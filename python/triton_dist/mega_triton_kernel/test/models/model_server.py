################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
import argparse
import os
import torch
import socket
import threading
import json
import traceback

from triton_dist.mega_triton_kernel import ModelBuilder
from triton_dist.mega_triton_kernel.models import DenseModel

from triton_dist.models import ModelConfig
from triton_dist.models.engine import AutoTokenizer
from triton_dist.models.utils import sample_token
from triton_dist.utils import (
    initialize_distributed, )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-32B", type=str, help="HuggingFace model name")
    parser.add_argument("--dtype", default="bfloat16", type=str, help="data type")
    parser.add_argument("--max_length", type=int, default=32768)
    parser.add_argument("--temperature", default=0.6, type=float)
    parser.add_argument("--top_p", default=0.95, type=float)
    parser.add_argument("--port", default=9999, type=int, help="Server port")

    return parser.parse_args()


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def generate_response(model, tokenizer, prompt_ids_tensor, args, RANK):
    """Generate response from the model using token IDs tensor on GPU"""
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    try:
        # prompt_ids_tensor is already a CUDA tensor with batch dimension
        input_ids = prompt_ids_tensor
        input_seq_len = input_ids.shape[1]
        output_ids = []
        gen_len = 1024  # Maximum generation length

        start_event.record()
        # Generation process
        for idx in range(gen_len + input_seq_len):
            model.kv_cache.inc_offset(1)
            if idx < input_seq_len:
                next_token = input_ids[:, idx].contiguous().reshape(-1, 1)
                logits = model.mega_forwrad(next_token)
                if idx == input_seq_len - 1:
                    next_token = sample_token(logits[:, -1, :], temperature=args.temperature, top_p=args.top_p)
            else:
                logits = model.mega_forwrad(next_token)
                next_token = sample_token(logits[:, -1, :], temperature=args.temperature, top_p=args.top_p)
            if idx >= input_seq_len - 1:
                output_ids.append(next_token)
                next_token_cpu = next_token[0, -1].cpu()
                if next_token_cpu.item() == model.eos_token_id:
                    break
        # Process output
        output_ids = torch.cat(output_ids, dim=1).cpu().tolist()
        end_event.record()
        start_event.wait()
        end_event.wait()
        torch.cuda.current_stream().synchronize()
        duration_ms = start_event.elapsed_time(end_event)

        # Decode output
        response = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

        if not response.strip():
            response = "I'm not sure how to answer that question."
            print("Warning: Generated response was empty, using default message.")

        return response, duration_ms
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        return None, None


def handle_client(client_socket, qwen3, tokenizer, args, RANK, WORLD_SIZE):
    """Handle client connection - keep connection open for multiple requests"""
    try:
        print(f"Client connected: {client_socket.getpeername()}")

        # Use a special token ID that is guaranteed to be invalid
        PAD_TOKEN_ID = -1

        while True:
            # Receive data from client
            data = client_socket.recv(4096).decode('utf-8')

            if not data:
                print(f"Client disconnected: {client_socket.getpeername()}")
                break

            # Parse JSON request
            try:
                request = json.loads(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding client request: {e}")
                error_response = json.dumps({"status": "error", "message": "Invalid request format"}).encode('utf-8')
                client_socket.send(error_response)
                continue

            prompt = request.get('prompt', '')

            if not prompt:
                error_response = json.dumps({"status": "error", "message": "Prompt is required"}).encode('utf-8')
                client_socket.send(error_response)
                continue

            print(f"Processing request from client {client_socket.getpeername()}")

            # Broadcast prompt tokens to all processes
            prompt_tensor = torch.full((1024, ), PAD_TOKEN_ID, dtype=torch.long).cuda()

            if RANK == 0:
                # Encode the prompt with chat template applied (only on rank 0)
                history = [{"role": "user", "content": prompt}]
                formatted_prompt = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
                prompt_ids = tokenizer.encode(formatted_prompt, max_length=1024, truncation=True)

                prompt_tensor[:len(prompt_ids)] = torch.tensor(prompt_ids).cuda()
                print(f"Encoded prompt tokens: {prompt_ids[:10]}... (total {len(prompt_ids)})")

            # Broadcast the prompt tensor to all processes
            torch.distributed.broadcast(prompt_tensor, src=0)

            # Efficiently determine valid length (find first padding token)
            valid_length = torch.count_nonzero(prompt_tensor != PAD_TOKEN_ID).item()

            # Slice the valid tokens directly on GPU (no CPU transfer)
            valid_prompt_ids = prompt_tensor[:valid_length].unsqueeze(0)  # Add batch dimension

            # Generate response using the sliced tensor directly on GPU
            response, duration_ms = generate_response(qwen3, tokenizer, valid_prompt_ids, args, RANK)
            # Only rank 0 needs to send the response
            if RANK == 0:
                if response is not None:
                    # Construct response JSON
                    response_data = {
                        'response': response,
                        'status': 'success',
                        'processing_time': duration_ms / 1000.0,
                    }
                    # Send response
                    client_socket.send(json.dumps(response_data).encode('utf-8'))
                    print("Response sent to client")
                else:
                    error_response = json.dumps({"status": "error", "message":
                                                 "Failed to generate response"}).encode('utf-8')
                    client_socket.send(error_response)

    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        # Close client connection
        try:
            client_socket.close()
            print(f"Client connection closed: {client_socket.getpeername()}")
        except Exception:
            pass


def main():
    args = parse_args()
    initialize_distributed(seed=0)
    RANK = int(os.environ.get("RANK", 0))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
    LOCAL_WORLD_SIZE = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    assert args.dtype == "bfloat16"

    dtype = DTYPE_MAP[args.dtype]
    model_config = ModelConfig(model_name=args.model, max_length=args.max_length, dtype=dtype, rank=RANK,
                               world_size=WORLD_SIZE)

    builder = ModelBuilder(rank=RANK, world_size=WORLD_SIZE, local_world_size=LOCAL_WORLD_SIZE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config)
    tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n<think>\n' }}{% endif %}"

    # Initialize model
    qwen3 = DenseModel(1, model_config, builder)  # Batch size set to 1

    if RANK == 0:
        # Create server socket
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Reuse address
        server.bind(('localhost', args.port))
        server.listen(5)
        print(f"Server started on port {args.port}, waiting for connections...")

        # Accept client connections
        while True:
            client_socket, addr = server.accept()

            # Create a thread for each client
            client_handler = threading.Thread(target=handle_client,
                                              args=(client_socket, qwen3, tokenizer, args, RANK, WORLD_SIZE))
            client_handler.daemon = True
            client_handler.start()
    else:
        # Non-rank 0 processes wait for prompts and participate in inference
        PAD_TOKEN_ID = -1

        print(f"Rank {RANK} waiting for prompts...")
        while True:
            # Create a placeholder to receive prompt tokens
            prompt_tensor = torch.full((1024, ), PAD_TOKEN_ID, dtype=torch.long).cuda()

            # Receive broadcasted prompt tokens
            torch.distributed.broadcast(prompt_tensor, src=0)

            # Efficiently determine valid length
            valid_length = torch.count_nonzero(prompt_tensor != PAD_TOKEN_ID).item()

            # Slice the valid tokens directly on GPU
            valid_prompt_ids = prompt_tensor[:valid_length].unsqueeze(0)  # Add batch dimension

            # Print first 10 tokens for debugging
            if valid_length > 0:
                print(
                    f"Rank {RANK} received prompt tokens: {valid_prompt_ids[0, :10].tolist()}... (total {valid_length})"
                )

            # Generate response using the sliced tensor
            response = generate_response(qwen3, tokenizer, valid_prompt_ids, args, RANK)  # noqa: F841


if __name__ == "__main__":
    main()
