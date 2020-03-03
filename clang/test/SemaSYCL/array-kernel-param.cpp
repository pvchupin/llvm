// RUN: %clang_cc1 -I %S/Inputs -fsycl-is-device -ast-dump %s | FileCheck %s

// This test checks that compiler generates correct kernel wrapper arguments for
// different accessors targets.

#include <sycl.hpp>

using namespace cl::sycl;

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel(Func kernelFunc) {
  kernelFunc();
}

int main() {

  using Accessor =
    accessor<int, 1, access::mode::read_write, access::target::global_buffer>;
    
  Accessor acc[2];
  int a[100];
  struct {
    Accessor member_acc[4];
  } struct_acc;

  kernel<class kernel_A>(
    [=]() {
    acc[1].use();
  });

  kernel<class kernel_B>(
    [=]() {
    int local = a[3];
  });

  kernel<class kernel_C>(
    [=]() {
    struct_acc.member_acc[2].use();
  });

}

// Check kernel_A parameters
// CHECK: FunctionDecl {{.*}}kernel_A{{.*}} 'void (__global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'cl::sycl::id<1>'
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init

// Check kernel_B parameters
// CHECK: FunctionDecl {{.*}}kernel_B{{.*}} 'void (wrapped_array)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'wrapped_array'
// CHECK: CallExpr {{.*}} 'void *'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void *(*)(void *, const void *, unsigned long) noexcept'
// CHECK-NEXT: DeclRefExpr {{.*}}__builtin_memcpy

// Check kernel_C parameters
// CHECK: FunctionDecl {{.*}}kernel_C{{.*}} 'void (struct {{.*}}, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>, __global int *, cl::sycl::range<1>, cl::sycl::range<1>, cl::sycl::id<1>)'
// CHECK-NEXT: ParmVarDecl {{.*}} 'struct {{.*}}'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::id<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc '__global int *'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::range<1>'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_member_acc 'cl::sycl::id<1>'
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}__init
