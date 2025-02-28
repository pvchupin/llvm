;__kernel void test_64(__global int* res)
;{
;	long tid = get_global_id(0);
;
;	switch(tid)
;	{
;	case 0:
;		res[tid] = 1;
;		break;
;	case 1:
;		res[tid] = 2;
;		break;
;	case 21474836481:
;		res[tid] = 3;
;		break;
;	}
;}
; bash$ clang -cc1 -triple spir64-unknown-unknown -x cl -cl-std=CL2.0 -O0 -include opencl.h -emit-llvm OpSwitch.cl -o test_64.ll

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r -emit-opaque-pointers %t.spv -o %t.bc
; RUN: llvm-dis < %t.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: 12 Switch {{[0-9]+}} {{[0-9]+}} 0 0 {{[0-9]+}} 1 0 {{[0-9]+}} 1 5 {{[0-9]+}}

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

;CHECK-LLVM: test_64
;CHECK-LLVM: entry
;CHECK-LLVM: switch i64 %6, label %sw.epilog [
;CHECK-LLVM: i64 0, label %sw.bb
;CHECK-LLVM: i64 1, label %sw.bb1
;CHECK-LLVM: i64 21474836481, label %sw.bb3

; Function Attrs: nounwind
define spir_kernel void @test_64(i32 addrspace(1)* %res) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2  !kernel_arg_type !3 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  %res.addr = alloca i32 addrspace(1)*, align 8
  %tid = alloca i64, align 8
  store i32 addrspace(1)* %res, i32 addrspace(1)** %res.addr, align 8
  %call = call spir_func i64 @_Z13get_global_idj(i32 0) #2
  store i64 %call, i64* %tid, align 8
  %0 = load i64, i64* %tid, align 8
  switch i64 %0, label %sw.epilog [
    i64 0, label %sw.bb
    i64 1, label %sw.bb1
    i64 21474836481, label %sw.bb3
  ]

;CHECK-LLVM: sw.bb
;CHECK-LLVM: preds = %entry
sw.bb:                                            ; preds = %entry
  %1 = load i64, i64* %tid, align 8
  %2 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx = getelementptr inbounds i32, i32 addrspace(1)* %2, i64 %1
  store i32 1, i32 addrspace(1)* %arrayidx, align 4
  br label %sw.epilog

;CHECK-LLVM: sw.bb1
;CHECK-LLVM: preds = %entry
sw.bb1:                                           ; preds = %entry
  %3 = load i64, i64* %tid, align 8
  %4 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx2 = getelementptr inbounds i32, i32 addrspace(1)* %4, i64 %3
  store i32 2, i32 addrspace(1)* %arrayidx2, align 4
  br label %sw.epilog

;CHECK-LLVM: sw.bb3
;CHECK-LLVM: preds = %entry
sw.bb3:                                           ; preds = %entry
  %5 = load i64, i64* %tid, align 8
  %6 = load i32 addrspace(1)*, i32 addrspace(1)** %res.addr, align 8
  %arrayidx4 = getelementptr inbounds i32, i32 addrspace(1)* %6, i64 %5
  store i32 3, i32 addrspace(1)* %arrayidx4, align 4
  br label %sw.epilog

;CHECK-LLVM: sw.epilog
;CHECK-LLVM: preds = %sw.bb3, %sw.bb1, %sw.bb, %entry
sw.epilog:                                        ; preds = %entry, %sw.bb3, %sw.bb1, %sw.bb
  ret void
}

; Function Attrs: nounwind
declare spir_func i64 @_Z13get_global_idj(i32) #1

attributes #0 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind }

!opencl.enable.FP_CONTRACT = !{}
!opencl.spir.version = !{!6}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!opencl.compiler.options = !{!8}
!llvm.ident = !{!9}

!1 = !{i32 1}
!2 = !{!"none"}
!3 = !{!"int*"}
!4 = !{!"int*"}
!5 = !{!""}
!6 = !{i32 1, i32 2}
!7 = !{i32 2, i32 0}
!8 = !{}
!9 = !{!"clang version 3.6.1"}
