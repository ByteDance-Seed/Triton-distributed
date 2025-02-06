target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

@__hip_cuid_1b8e99b24a418f00 = addrspace(1) global i8 0
@llvm.compiler.used = appending addrspace(1) global [29 x ptr] [ptr @_Z18load_acquire_agentPU3AS1m, ptr @_Z18load_relaxed_agentPU3AS1m, ptr @_Z19load_acquire_systemPU3AS1i, ptr @_Z19load_relaxed_systemPU3AS1i, ptr @_Z19store_relaxed_agentPU3AS1m, ptr @_Z19store_release_agentPU3AS1m, ptr @_Z20store_relaxed_systemPU3AS1ii, ptr @_Z20store_release_systemPU3AS1ii, ptr @_Z21atom_add_acqrel_agentPU3AS1iS0_, ptr @_Z21red_add_release_agentPU3AS1iS0_, ptr @_Z22atom_add_acqrel_systemPU3AS1ii, ptr @_Z22atom_add_acquire_agentPU3AS1iS0_, ptr @_Z22atom_add_relaxed_agentPU3AS1iS0_, ptr @_Z22load_acquire_workgroupPU3AS1m, ptr @_Z22load_relaxed_workgroupPU3AS1m, ptr @_Z22red_add_release_systemPU3AS1ii, ptr @_Z23atom_add_acquire_systemPU3AS1ii, ptr @_Z23atom_add_relaxed_systemPU3AS1ii, ptr @_Z23store_relaxed_workgroupPU3AS1m, ptr @_Z23store_release_workgroupPU3AS1m, ptr @_Z29atom_cas_acqrel_relaxed_agentPU3AS1iS0_S0_, ptr @_Z30atom_cas_acqrel_relaxed_systemPU3AS1iS0_i, ptr @_Z30atom_cas_acquire_relaxed_agentPU3AS1iS0_S0_, ptr @_Z30atom_cas_relaxed_relaxed_agentPU3AS1iS0_S0_, ptr @_Z30atom_cas_release_relaxed_agentPU3AS1iS0_S0_, ptr @_Z31atom_cas_acquire_relaxed_systemPU3AS1iS0_i, ptr @_Z31atom_cas_relaxed_relaxed_systemPU3AS1iS0_i, ptr @_Z31atom_cas_release_relaxed_systemPU3AS1iS0_i, ptr addrspacecast (ptr addrspace(1) @__hip_cuid_1b8e99b24a418f00 to ptr)], section "llvm.metadata"

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z22load_acquire_workgroupPU3AS1m(ptr addrspace(1) nocapture noundef readonly %0) #0 {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("workgroup-one-as") acquire, align 8
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z22load_relaxed_workgroupPU3AS1m(ptr addrspace(1) nocapture noundef readonly %0) #0 {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 8
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z18load_acquire_agentPU3AS1m(ptr addrspace(1) nocapture noundef readonly %0) #0 {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("agent-one-as") acquire, align 8
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z18load_relaxed_agentPU3AS1m(ptr addrspace(1) nocapture noundef readonly %0) #0 {
  %2 = load atomic i64, ptr addrspace(1) %0 syncscope("agent-one-as") monotonic, align 8
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z19load_acquire_systemPU3AS1i(ptr addrspace(1) nocapture noundef readonly %0) #0 {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("one-as") acquire, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z19load_relaxed_systemPU3AS1i(ptr addrspace(1) nocapture noundef readonly %0) #0 {
  %2 = load atomic i32, ptr addrspace(1) %0 syncscope("one-as") monotonic, align 4
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z23store_release_workgroupPU3AS1m(ptr addrspace(1) nocapture noundef writeonly %0) #0 {
  store atomic i64 1, ptr addrspace(1) %0 syncscope("workgroup-one-as") release, align 8
  ret i64 1
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z23store_relaxed_workgroupPU3AS1m(ptr addrspace(1) nocapture noundef writeonly %0) #0 {
  store atomic i64 1, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 8
  ret i64 1
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z19store_release_agentPU3AS1m(ptr addrspace(1) nocapture noundef writeonly %0) #0 {
  store atomic i64 1, ptr addrspace(1) %0 syncscope("agent-one-as") release, align 8
  ret i64 1
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i64 @_Z19store_relaxed_agentPU3AS1m(ptr addrspace(1) nocapture noundef writeonly %0) #0 {
  store atomic i64 1, ptr addrspace(1) %0 syncscope("agent-one-as") monotonic, align 8
  ret i64 1
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z20store_release_systemPU3AS1ii(ptr addrspace(1) nocapture noundef writeonly %0, i32 noundef returned %1) #0 {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("one-as") release, align 4
  ret i32 %1
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z20store_relaxed_systemPU3AS1ii(ptr addrspace(1) nocapture noundef writeonly %0, i32 noundef returned %1) #0 {
  store atomic i32 %1, ptr addrspace(1) %0 syncscope("one-as") monotonic, align 4
  ret i32 %1
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z21red_add_release_agentPU3AS1iS0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef readonly %1) #0 {
  %3 = load i32, ptr addrspace(1) %1, align 4, !tbaa !6
  %4 = atomicrmw add ptr addrspace(1) %0, i32 %3 syncscope("agent-one-as") release, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z22red_add_release_systemPU3AS1ii(ptr addrspace(1) nocapture noundef %0, i32 noundef %1) #0 {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("one-as") release, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z22atom_add_acquire_agentPU3AS1iS0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef readonly %1) #0 {
  %3 = load i32, ptr addrspace(1) %1, align 4, !tbaa !6
  %4 = atomicrmw add ptr addrspace(1) %0, i32 %3 syncscope("agent-one-as") acquire, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z22atom_add_relaxed_agentPU3AS1iS0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef readonly %1) #0 {
  %3 = load i32, ptr addrspace(1) %1, align 4, !tbaa !6
  %4 = atomicrmw add ptr addrspace(1) %0, i32 %3 syncscope("agent-one-as") monotonic, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z21atom_add_acqrel_agentPU3AS1iS0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef readonly %1) #0 {
  %3 = load i32, ptr addrspace(1) %1, align 4, !tbaa !6
  %4 = atomicrmw add ptr addrspace(1) %0, i32 %3 syncscope("agent-one-as") acq_rel, align 4
  ret i32 %4
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z23atom_add_acquire_systemPU3AS1ii(ptr addrspace(1) nocapture noundef %0, i32 noundef %1) #0 {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("one-as") acquire, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z23atom_add_relaxed_systemPU3AS1ii(ptr addrspace(1) nocapture noundef %0, i32 noundef %1) #0 {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("one-as") monotonic, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z22atom_add_acqrel_systemPU3AS1ii(ptr addrspace(1) nocapture noundef %0, i32 noundef %1) #0 {
  %3 = atomicrmw add ptr addrspace(1) %0, i32 %1 syncscope("one-as") acq_rel, align 4
  ret i32 %3
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z30atom_cas_acquire_relaxed_agentPU3AS1iS0_S0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, ptr addrspace(1) nocapture noundef readonly %2) #0 {
  %4 = load i32, ptr addrspace(1) %2, align 4, !tbaa !6
  %5 = load i32, ptr addrspace(1) %1, align 4
  %6 = cmpxchg ptr addrspace(1) %0, i32 %5, i32 %4 syncscope("agent-one-as") acquire monotonic, align 4
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, ptr addrspace(1) %1, align 4
  br label %10

10:                                               ; preds = %8, %3
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z30atom_cas_release_relaxed_agentPU3AS1iS0_S0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, ptr addrspace(1) nocapture noundef readonly %2) #0 {
  %4 = load i32, ptr addrspace(1) %2, align 4, !tbaa !6
  %5 = load i32, ptr addrspace(1) %1, align 4
  %6 = cmpxchg ptr addrspace(1) %0, i32 %5, i32 %4 syncscope("agent-one-as") release monotonic, align 4
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, ptr addrspace(1) %1, align 4
  br label %10

10:                                               ; preds = %8, %3
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z30atom_cas_relaxed_relaxed_agentPU3AS1iS0_S0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, ptr addrspace(1) nocapture noundef readonly %2) #0 {
  %4 = load i32, ptr addrspace(1) %2, align 4, !tbaa !6
  %5 = load i32, ptr addrspace(1) %1, align 4
  %6 = cmpxchg ptr addrspace(1) %0, i32 %5, i32 %4 syncscope("agent-one-as") monotonic monotonic, align 4
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, ptr addrspace(1) %1, align 4
  br label %10

10:                                               ; preds = %8, %3
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z29atom_cas_acqrel_relaxed_agentPU3AS1iS0_S0_(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, ptr addrspace(1) nocapture noundef readonly %2) #0 {
  %4 = load i32, ptr addrspace(1) %2, align 4, !tbaa !6
  %5 = load i32, ptr addrspace(1) %1, align 4
  %6 = cmpxchg ptr addrspace(1) %0, i32 %5, i32 %4 syncscope("agent-one-as") acq_rel monotonic, align 4
  %7 = extractvalue { i32, i1 } %6, 1
  br i1 %7, label %10, label %8

8:                                                ; preds = %3
  %9 = extractvalue { i32, i1 } %6, 0
  store i32 %9, ptr addrspace(1) %1, align 4
  br label %10

10:                                               ; preds = %8, %3
  %11 = zext i1 %7 to i32
  ret i32 %11
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z31atom_cas_acquire_relaxed_systemPU3AS1iS0_i(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, i32 noundef %2) #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4
  %5 = cmpxchg ptr addrspace(1) %0, i32 %4, i32 %2 syncscope("one-as") acquire monotonic, align 4
  %6 = extractvalue { i32, i1 } %5, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %3
  %8 = extractvalue { i32, i1 } %5, 0
  store i32 %8, ptr addrspace(1) %1, align 4
  br label %9

9:                                                ; preds = %7, %3
  %10 = zext i1 %6 to i32
  ret i32 %10
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z31atom_cas_release_relaxed_systemPU3AS1iS0_i(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, i32 noundef %2) #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4
  %5 = cmpxchg ptr addrspace(1) %0, i32 %4, i32 %2 syncscope("one-as") release monotonic, align 4
  %6 = extractvalue { i32, i1 } %5, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %3
  %8 = extractvalue { i32, i1 } %5, 0
  store i32 %8, ptr addrspace(1) %1, align 4
  br label %9

9:                                                ; preds = %7, %3
  %10 = zext i1 %6 to i32
  ret i32 %10
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z31atom_cas_relaxed_relaxed_systemPU3AS1iS0_i(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, i32 noundef %2) #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4
  %5 = cmpxchg ptr addrspace(1) %0, i32 %4, i32 %2 syncscope("one-as") monotonic monotonic, align 4
  %6 = extractvalue { i32, i1 } %5, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %3
  %8 = extractvalue { i32, i1 } %5, 0
  store i32 %8, ptr addrspace(1) %1, align 4
  br label %9

9:                                                ; preds = %7, %3
  %10 = zext i1 %6 to i32
  ret i32 %10
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite)
define internal noundef i32 @_Z30atom_cas_acqrel_relaxed_systemPU3AS1iS0_i(ptr addrspace(1) nocapture noundef %0, ptr addrspace(1) nocapture noundef %1, i32 noundef %2) #0 {
  %4 = load i32, ptr addrspace(1) %1, align 4
  %5 = cmpxchg ptr addrspace(1) %0, i32 %4, i32 %2 syncscope("one-as") acq_rel monotonic, align 4
  %6 = extractvalue { i32, i1 } %5, 1
  br i1 %6, label %9, label %7

7:                                                ; preds = %3
  %8 = extractvalue { i32, i1 } %5, 0
  store i32 %8, ptr addrspace(1) %1, align 4
  br label %9

9:                                                ; preds = %7, %3
  %10 = zext i1 %6 to i32
  ret i32 %10
}

attributes #0 = { mustprogress nofree norecurse nounwind willreturn memory(argmem: readwrite) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx942" "target-features"="+16-bit-insts,+atomic-buffer-global-pk-add-f16-insts,+atomic-ds-pk-add-16-insts,+atomic-fadd-rtn-insts,+atomic-flat-pk-add-16-insts,+atomic-global-pk-add-bf16-inst,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+fp8-conversion-insts,+fp8-insts,+gfx8-insts,+gfx9-insts,+gfx90a-insts,+gfx940-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64,-tgsplit" }

!llvm.module.flags = !{!0, !1, !2, !3}
!opencl.ocl.version = !{!4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"amdgpu_code_object_version", i32 500}
!1 = !{i32 1, !"amdgpu_printf_kind", !"hostcall"}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 8, !"PIC Level", i32 2}
!4 = !{i32 2, i32 0}
!5 = !{!"AMD clang version 18.0.0git (https://github.com/RadeonOpenCompute/llvm-project roc-6.3.0 24455 f24aa3b4a91f6ee2fcd15629ba0b49fa545d8d6b)"}
!6 = !{!7, !7, i64 0}
!7 = !{!"int", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C++ TBAA"}
