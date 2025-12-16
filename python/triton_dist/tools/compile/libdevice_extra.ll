; ModuleID = './3rdparty/triton/third_party/amd/backend/lib/libdevice_extra.ll'
source_filename = "llvm-link"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128:128:48-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn
; compile this with `hipcc -S -emit-llvm __gen_smid.hip -o -`
;  __global__ void __extra_smid(int *ptr) {   ptr[0] = __smid(); }
define linkonce hidden i32 @__extra_smid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 2884)
  %3 = tail call i32 @llvm.amdgcn.s.getreg(i32 6164)
  %4 = tail call i32 @llvm.amdgcn.s.getreg(i32 6660)
  %5 = shl i32 %3, 6
  %6 = shl i32 %2, 4
  %7 = or i32 %5, %6
  %8 = or i32 %7, %4
  ret i32 %8
}

define linkonce hidden i32 @__extra_seid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 2884)
  ret i32 %2
}

define linkonce hidden i32 @__extra_xccid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 6164)
  ret i32 %2
}

define linkonce hidden i32 @__extra_cuid() local_unnamed_addr #3 {
  %2 = tail call i32 @llvm.amdgcn.s.getreg(i32 6660)
  ret i32 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define linkonce hidden i64 @__extra_clock() local_unnamed_addr #3 {
  %2 = tail call noundef i64 @llvm.amdgcn.s.memtime()
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define linkonce hidden i64 @__extra_wallclock() local_unnamed_addr #3 {
  %2 = tail call noundef i64 @llvm.amdgcn.s.memrealtime()
  ret i64 %2
}

; Function Attrs: mustprogress nofree norecurse nounwind willreturn
define linkonce hidden void @__extra_fence_system() local_unnamed_addr {
  fence syncscope("agent") release
  ret void
}

define linkonce hidden void @__extra_fence_acquire_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") acquire
  ret void
}

define linkonce hidden void @__extra_fence_acquire_agent() local_unnamed_addr {
  fence syncscope("agent") acquire
  ret void
}

define linkonce hidden void @__extra_fence_acquire_system() local_unnamed_addr {
  fence syncscope("system") acquire
  ret void
}

define linkonce hidden void @__extra_fence_release_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") release
  ret void
}

define linkonce hidden void @__extra_fence_release_agent() local_unnamed_addr {
  fence syncscope("agent") release
  ret void
}

define linkonce hidden void @__extra_fence_release_system() local_unnamed_addr {
  fence syncscope("system") release
  ret void
}

define linkonce hidden void @__extra_fence_acq_rel_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") acq_rel
  ret void
}

define linkonce hidden void @__extra_fence_acq_rel_agent() local_unnamed_addr {
  fence syncscope("agent") acq_rel
  ret void
}

define linkonce hidden void @__extra_fence_acq_rel_system() local_unnamed_addr {
  fence syncscope("system") acq_rel
  ret void
}

define linkonce hidden void @__extra_fence_seq_cst_workgroup() local_unnamed_addr {
  fence syncscope("workgroup") seq_cst
  ret void
}

define linkonce hidden void @__extra_fence_seq_cst_agent() local_unnamed_addr {
  fence syncscope("agent") seq_cst
  ret void
}

define linkonce hidden void @__extra_fence_seq_cst_system() local_unnamed_addr {
  fence syncscope("system") seq_cst
  ret void
}

attributes #3 = {
  nounwind
  "target-cpu"="gfx942"
  "uniform-work-group-size"="true"
  "target-features"="+gfx940-insts"
}