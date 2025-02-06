module {
  tt.func public @kernel_consumer_gemm(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}, %arg4: i32, %arg5: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg6: i32 {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}, %arg9: i32 {tt.divisibility = 16 : i32}, %arg10: i32 {tt.divisibility = 16 : i32}, %arg11: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c31_i32 = arith.constant 31 : i32
    %c127_i32 = arith.constant 127 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<32x128xf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<32> : tensor<128x32xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<128x128xf32>
    %c32_i32 = arith.constant 32 : i32
    %c128_i32 = arith.constant 128 : i32
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.addi %arg6, %c127_i32 : i32
    %2 = arith.divsi %1, %c128_i32 : i32
    %3 = arith.addi %arg7, %c127_i32 : i32
    %4 = arith.divsi %3, %c128_i32 : i32
    %5 = arith.muli %4, %c8_i32 : i32
    %6 = arith.divsi %0, %5 : i32
    %7 = arith.muli %6, %c8_i32 : i32
    %8 = arith.subi %2, %7 : i32
    %9 = arith.minsi %8, %c8_i32 : i32
    %10 = arith.remsi %0, %5 : i32
    %11 = arith.remsi %10, %9 : i32
    %12 = arith.addi %7, %11 : i32
    %13 = arith.divsi %10, %9 : i32
    %14 = arith.muli %12, %c128_i32 : i32
    %15 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %16 = tt.splat %14 : i32 -> tensor<128xi32>
    %17 = arith.addi %16, %15 : tensor<128xi32>
    %18 = tt.splat %arg6 : i32 -> tensor<128xi32>
    %19 = arith.remsi %17, %18 : tensor<128xi32>
    %20 = arith.muli %13, %c128_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<128xi32>
    %22 = arith.addi %21, %15 : tensor<128xi32>
    %23 = tt.splat %arg7 : i32 -> tensor<128xi32>
    %24 = arith.remsi %22, %23 : tensor<128xi32>
    %25 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %26 = tt.expand_dims %19 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %27 = tt.splat %arg9 : i32 -> tensor<128x1xi32>
    %28 = arith.muli %26, %27 : tensor<128x1xi32>
    %29 = tt.expand_dims %25 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %30 = tt.broadcast %28 : tensor<128x1xi32> -> tensor<128x32xi32>
    %31 = tt.broadcast %29 : tensor<1x32xi32> -> tensor<128x32xi32>
    %32 = arith.addi %30, %31 : tensor<128x32xi32>
    %33 = tt.splat %arg0 : !tt.ptr<f16> -> tensor<128x32x!tt.ptr<f16>>
    %34 = tt.addptr %33, %32 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
    %35 = tt.expand_dims %25 {axis = 1 : i32} : tensor<32xi32> -> tensor<32x1xi32>
    %36 = tt.splat %arg10 : i32 -> tensor<32x1xi32>
    %37 = arith.muli %35, %36 : tensor<32x1xi32>
    %38 = tt.expand_dims %24 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %39 = tt.broadcast %37 : tensor<32x1xi32> -> tensor<32x128xi32>
    %40 = tt.broadcast %38 : tensor<1x128xi32> -> tensor<32x128xi32>
    %41 = arith.addi %39, %40 : tensor<32x128xi32>
    %42 = tt.splat %arg1 : !tt.ptr<f16> -> tensor<32x128x!tt.ptr<f16>>
    %43 = tt.addptr %42, %41 : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
    %44 = distributed.wait %34, %arg5, scope = gpu semantic = acquire : tensor<128x32x!tt.ptr<f16>>, !tt.ptr<i32>
    %45 = arith.addi %arg8, %c31_i32 : i32
    %46 = arith.divsi %45, %c32_i32 : i32
    %47 = arith.muli %arg10, %c32_i32 : i32
    %48 = tt.splat %47 : i32 -> tensor<32x128xi32>
    %49:3 = scf.for %arg12 = %c0_i32 to %46 step %c1_i32 iter_args(%arg13 = %cst_2, %arg14 = %44, %arg15 = %43) -> (tensor<128x128xf32>, tensor<128x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>)  : i32 {
      %67 = arith.muli %arg12, %c32_i32 : i32
      %68 = arith.subi %arg8, %67 : i32
      %69 = tt.splat %68 : i32 -> tensor<1x32xi32>
      %70 = arith.cmpi slt, %29, %69 : tensor<1x32xi32>
      %71 = tt.broadcast %70 : tensor<1x32xi1> -> tensor<128x32xi1>
      %72 = tt.load %arg14, %71, %cst_0 : tensor<128x32x!tt.ptr<f16>>
      %73 = tt.splat %68 : i32 -> tensor<32x1xi32>
      %74 = arith.cmpi slt, %35, %73 : tensor<32x1xi32>
      %75 = tt.broadcast %74 : tensor<32x1xi1> -> tensor<32x128xi1>
      %76 = tt.load %arg15, %75, %cst : tensor<32x128x!tt.ptr<f16>>
      %77 = tt.dot %72, %76, %arg13, inputPrecision = tf32 : tensor<128x32xf16> * tensor<32x128xf16> -> tensor<128x128xf32>
      %78 = tt.addptr %arg14, %cst_1 : tensor<128x32x!tt.ptr<f16>>, tensor<128x32xi32>
      %79 = tt.addptr %arg15, %48 : tensor<32x128x!tt.ptr<f16>>, tensor<32x128xi32>
      scf.yield %77, %78, %79 : tensor<128x128xf32>, tensor<128x32x!tt.ptr<f16>>, tensor<32x128x!tt.ptr<f16>>
    }
    %50 = arith.truncf %49#0 : tensor<128x128xf32> to tensor<128x128xf16>
    %51 = tt.expand_dims %17 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %52 = tt.splat %arg11 : i32 -> tensor<128x1xi32>
    %53 = arith.muli %52, %51 : tensor<128x1xi32>
    %54 = tt.splat %arg2 : !tt.ptr<f16> -> tensor<128x1x!tt.ptr<f16>>
    %55 = tt.addptr %54, %53 : tensor<128x1x!tt.ptr<f16>>, tensor<128x1xi32>
    %56 = tt.expand_dims %22 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %57 = tt.broadcast %55 : tensor<128x1x!tt.ptr<f16>> -> tensor<128x128x!tt.ptr<f16>>
    %58 = tt.broadcast %56 : tensor<1x128xi32> -> tensor<128x128xi32>
    %59 = tt.addptr %57, %58 : tensor<128x128x!tt.ptr<f16>>, tensor<128x128xi32>
    %60 = tt.splat %arg6 : i32 -> tensor<128x1xi32>
    %61 = arith.cmpi slt, %51, %60 : tensor<128x1xi32>
    %62 = tt.splat %arg7 : i32 -> tensor<1x128xi32>
    %63 = arith.cmpi slt, %56, %62 : tensor<1x128xi32>
    %64 = tt.broadcast %61 : tensor<128x1xi1> -> tensor<128x128xi1>
    %65 = tt.broadcast %63 : tensor<1x128xi1> -> tensor<128x128xi1>
    %66 = arith.andi %64, %65 : tensor<128x128xi1>
    tt.store %59, %50, %66 : tensor<128x128x!tt.ptr<f16>>
    tt.return
  }
}