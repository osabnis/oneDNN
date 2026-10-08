/*******************************************************************************
* Copyright 2026 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <iomanip>
#include <vector>

#include "oneapi/dnnl/dnnl_debug.h"

#include "common/utils.hpp"

#include "cpu/x64/ir/dump.hpp"

namespace dnnl {
namespace impl {
namespace cpu {
namespace x64 {
namespace ir {

namespace {

// Returns the name of an operation kind, which is its `op_kind_t` enumerator
// name. The switch has no `default`, so a new kind triggers a `-Wswitch`
// warning until it is handled here.
const char *kind_name(op_kind_t kind) {
    switch (kind) {
        case op_kind_t::mov_imm: return "mov_imm";
        case op_kind_t::mov_reg: return "mov_reg";
        case op_kind_t::add_imm: return "add_imm";
        case op_kind_t::add_reg: return "add_reg";
        case op_kind_t::load: return "load";
        case op_kind_t::vzero: return "vzero";
        case op_kind_t::vload: return "vload";
        case op_kind_t::vstore: return "vstore";
        case op_kind_t::vload_scalar: return "vload_scalar";
        case op_kind_t::vstore_scalar: return "vstore_scalar";
        case op_kind_t::vload_bcast: return "vload_bcast";
        case op_kind_t::vdot: return "vdot";
        case op_kind_t::vadd: return "vadd";
        case op_kind_t::vmul: return "vmul";
        case op_kind_t::vhreduce: return "vhreduce";
        case op_kind_t::inject_postops: return "inject_postops";
        case op_kind_t::set_mask_imm: return "set_mask_imm";
        case op_kind_t::vload_masked: return "vload_masked";
        case op_kind_t::vstore_masked: return "vstore_masked";
        case op_kind_t::prefetch: return "prefetch";
        case op_kind_t::loop_begin: return "loop_begin";
        case op_kind_t::loop_end: return "loop_end";
        case op_kind_t::label: return "label";
        case op_kind_t::jmp: return "jmp";
        case op_kind_t::jz: return "jz";
    }
    assert(!"unknown op kind");
    return "?";
}

const char *reg_kind_name(reg_kind_t kind) {
    switch (kind) {
        case reg_kind_t::gpr: return "gpr";
        case reg_kind_t::vec: return "vec";
        case reg_kind_t::mask: return "mask";
    }
    assert(!"unknown reg kind");
    return "?";
}

bool is_valid(const ir_t &ir, vreg_t v) {
    return (int)v >= 0 && (int)v < ir.n_vregs();
}

// Returns the name of the physical register `phys` for a vreg of kind `kind`.
// The register type is the one the emitter uses. The AVX-512 backend keeps a
// vec in a `zmm` and a mask in a `k` register. The AVX2 backend keeps both in a
// `ymm`.
std::string phys_str(cpu_isa_t isa, reg_kind_t kind, int phys) {
    const bool is_avx512 = is_superset(isa, avx512_core);
    switch (kind) {
        case reg_kind_t::gpr: return Xbyak::Reg64(phys).toString();
        case reg_kind_t::vec:
            return is_avx512 ? Xbyak::Zmm(phys).toString()
                             : Xbyak::Ymm(phys).toString();
        case reg_kind_t::mask:
            return is_avx512 ? Xbyak::Opmask(phys).toString()
                             : Xbyak::Ymm(phys).toString();
    }
    assert(!"unknown reg kind");
    return "?";
}

// Returns where the allocation in `info` keeps `v`: a physical register, or a
// stack slot `[rsp+<off>]`, the address that the emitter uses. `temps` are the
// temps of the operation that has `v` as an operand, or null outside an
// operation. A spilled operand is followed by its temp, `(temp <reg>)`, or by
// `(no temp)` when the operation has no temp for it.
std::string location_str(const ir_t &ir, const kernel_info_t &info, vreg_t v,
        const std::vector<temp_reg_t> *temps) {
    const assignment_t &a = info.alloc->assignments[(int)v];
    const reg_kind_t kind = ir.vreg_info()[(int)v].kind;
    if (!a.spilled) return a.phys < 0 ? "?" : phys_str(info.isa, kind, a.phys);

    const std::string slot = "[rsp+" + std::to_string(a.slot) + "]";
    if (!temps) return slot;
    for (const temp_reg_t &t : *temps)
        if (t.vreg == v)
            return slot + "(temp " + phys_str(info.isa, kind, t.phys) + ")";
    return slot + "(no temp)";
}

// Returns `g<id>`, `<dt>:v<id>`, or `m<id>` by the kind of `v`. A vec vreg
// carries its data type, so the reader does not have to look it up. An id
// outside the IR prints as `?<id>` rather than failing, because the IR dump is
// most useful on an IR that is suspected to be wrong. When `info` is not null,
// the location of `v` follows after `@` (see `location_str()`).
std::string vreg_str(const ir_t &ir, vreg_t v,
        const kernel_info_t *info = nullptr,
        const std::vector<temp_reg_t> *temps = nullptr) {
    if (v == vreg_t::none) return "none";
    const std::string id = std::to_string((int)v);
    if (!is_valid(ir, v)) return "?" + id;
    const std::string at = info ? "@" + location_str(ir, *info, v, temps) : "";
    switch (ir.vreg_info()[(int)v].kind) {
        case reg_kind_t::gpr: return "g" + id + at;
        case reg_kind_t::vec:
            return std::string(dnnl_dt2str(ir.vreg_info()[(int)v].dt)) + ":v"
                    + id + at;
        case reg_kind_t::mask: return "m" + id + at;
    }
    assert(!"unknown reg kind");
    return "?" + id;
}

// Returns the name of register file `f`: the name of the first kind that
// allocates from it. On AVX2, a mask is a vector register, so masks allocate
// from the file `vec`.
std::string file_name(const reg_pools_t &pools, int f) {
    for (int k = 0; k < (int)pools.kind_to_file.size(); k++)
        if (pools.kind_to_file[k] == f) return reg_kind_name((reg_kind_t)k);
    return "file" + std::to_string(f);
}

std::string label_str(label_t l) {
    return "L" + std::to_string((int)l);
}

// Returns the memory operand of `op`. `data` is the vec vreg the access moves,
// or `none` for an access that moves no vec value (`load`, `prefetch`). A
// vector access is prefixed with the data type in memory, the same way a vec
// vreg is prefixed with its data type. A converting access therefore shows two
// different types (`vload f32:v3, bf16:[g0+0]`).
std::string mem_str(const ir_t &ir, const op_t &op, vreg_t data,
        const kernel_info_t *info, const std::vector<temp_reg_t> *temps) {
    std::string s;
    if (data != vreg_t::none && op.mem_dt != data_type::undef) {
        s += dnnl_dt2str(op.mem_dt);
        s += ":";
    }
    s += "[";
    s += op.mem.is_param ? "param" : vreg_str(ir, op.mem.base, info, temps);
    if (op.mem.disp >= 0) s += "+";
    s += std::to_string(op.mem.disp);
    s += "]";
    return s;
}

// Returns the operands of an `inject_postops` op from its side-table entry:
// the accumulators, then `base=` unless the base pointer is `none`, then
// `off=` unless the offsets are empty.
std::string postops_str(const ir_t &ir, const op_t &op,
        const kernel_info_t *info, const std::vector<temp_reg_t> *temps) {
    const auto &table = ir.inject_postops_args();
    if (op.imm < 0 || op.imm >= (dim_t)table.size())
        return "args=?" + std::to_string(op.imm);
    const inject_postops_args_t &args = table[(int)op.imm];

    std::string s;
    for (size_t i = 0; i < args.acc.size(); i++) {
        if (i > 0) s += ", ";
        s += vreg_str(ir, args.acc[i], info, temps);
    }
    if (args.base_ptr != vreg_t::none)
        s += ", base=" + vreg_str(ir, args.base_ptr, info, temps);
    if (!args.out_byte_off.empty()) {
        s += ", off=[";
        for (size_t i = 0; i < args.out_byte_off.size(); i++) {
            if (i > 0) s += ", ";
            s += std::to_string(args.out_byte_off[i]);
        }
        s += "]";
    }
    return s;
}

// Returns the text of one operation, without index or indentation. Every kind
// prints as its name followed by its operands, except the control-flow
// markers. A loop prints as an opening and a closing brace, and a label prints
// as `L<id>:`. When `info` is not null, each vreg shows its location, and
// `temps` are the temps of the operation.
std::string op_str(const ir_t &ir, const op_t &op, const kernel_info_t *info,
        const std::vector<temp_reg_t> *temps) {
    const auto r = [&](vreg_t v) { return vreg_str(ir, v, info, temps); };
    const auto mem
            = [&](vreg_t data) { return mem_str(ir, op, data, info, temps); };
    const std::string k = std::string(kind_name(op.kind)) + " ";

    switch (op.kind) {
        case op_kind_t::mov_imm:
        case op_kind_t::add_imm:
        case op_kind_t::set_mask_imm:
            return k + r(op.dst) + ", " + std::to_string(op.imm);
        case op_kind_t::mov_reg:
        case op_kind_t::add_reg:
        case op_kind_t::vadd:
        case op_kind_t::vmul:
        case op_kind_t::vhreduce: return k + r(op.dst) + ", " + r(op.s0);
        case op_kind_t::load: return k + r(op.dst) + ", " + mem(vreg_t::none);
        case op_kind_t::vzero: return k + r(op.dst);
        case op_kind_t::vload:
        case op_kind_t::vload_scalar:
        case op_kind_t::vload_bcast: return k + r(op.dst) + ", " + mem(op.dst);
        case op_kind_t::vstore:
        case op_kind_t::vstore_scalar: return k + mem(op.s0) + ", " + r(op.s0);
        case op_kind_t::vdot:
            return k + r(op.dst) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::inject_postops:
            return k + postops_str(ir, op, info, temps);
        case op_kind_t::vload_masked:
            return k + r(op.dst) + ", " + mem(op.dst) + ", " + r(op.s1);
        case op_kind_t::vstore_masked:
            return k + mem(op.s0) + ", " + r(op.s0) + ", " + r(op.s1);
        case op_kind_t::prefetch: return k + mem(vreg_t::none);
        case op_kind_t::loop_begin:
            return "loop " + r(op.dst) + " = "
                    + (op.init_is_reg ? r(op.s0) : std::to_string(op.imm))
                    + " {";
        case op_kind_t::loop_end:
            return "} // " + r(op.dst) + " -= 1, repeat while > 0";
        case op_kind_t::label: return label_str(op.label_id) + ":";
        case op_kind_t::jmp: return k + label_str(op.label_id);
        case op_kind_t::jz: return k + r(op.s0) + ", " + label_str(op.label_id);
    }
    assert(!"unknown op kind");
    return k;
}

// Returns `<value> at op <index>` for the first largest value in `per_op`, or
// `0` when every value is 0.
std::string max_at_str(const std::vector<int> &per_op) {
    const auto it = std::max_element(per_op.begin(), per_op.end());
    if (it == per_op.end() || *it == 0) return "0";
    return std::to_string(*it) + " at op "
            + std::to_string((int)(it - per_op.begin()));
}

int file_of(const ir_t &ir, const reg_pools_t &pools, int v) {
    return pools.kind_to_file[(int)ir.vreg_info()[v].kind];
}

// Returns the vregs that allocate from file `f`.
std::vector<int> file_vregs(const ir_t &ir, const reg_pools_t &pools, int f) {
    std::vector<int> vregs;
    for (int v = 0; v < ir.n_vregs(); v++)
        if (file_of(ir, pools, v) == f) vregs.push_back(v);
    return vregs;
}

// Returns the register pressure of each register file at each operation,
// indexed [file][op]: the vregs of the file that need a register at the
// operation. A vreg needs a register at operation `i` when it is live on entry
// to `i`, or when `i` writes it. A write needs a register even when the value
// is never read.
std::vector<std::vector<int>> count_pressure(
        const ir_t &ir, const reg_pools_t &pools) {
    std::vector<std::vector<int8_t>> live_in;
    compute_liveness(ir, live_in);

    std::vector<std::vector<int>> pressure(
            pools.files.size(), std::vector<int>(ir.n_ops(), 0));
    std::vector<int> defs, uses;
    for (int i = 0; i < ir.n_ops(); i++) {
        std::vector<int8_t> needs = live_in[i];
        ir.def_use(ir.ops()[i], defs, uses);
        for (int v : defs)
            needs[v] = 1;
        for (int v = 0; v < ir.n_vregs(); v++)
            if (needs[v]) pressure[file_of(ir, pools, v)][i]++;
    }
    return pressure;
}

int n_spilled(const ir_t &ir, const kernel_info_t &info, int f) {
    int n = 0;
    for (int v : file_vregs(ir, info.reg_cfg->pools, f))
        if (info.alloc->assignments[v].spilled) n++;
    return n;
}

// Returns the lines of the register allocation, one line per register file.
// The peak is the largest register pressure of the file.
std::string alloc_lines(const ir_t &ir, const kernel_info_t &info,
        const std::vector<std::vector<int>> &pressure) {
    const reg_pools_t &pools = info.reg_cfg->pools;
    ostringstream_t ss;
    for (int f = 0; f < (int)pools.files.size(); f++)
        ss << "alloc " << file_name(pools, f) << ": pool "
           << pools.files[f].regs.size() << ", peak " << max_at_str(pressure[f])
           << ", spilled " << n_spilled(ir, info, f) << "\n";
    return ss.str();
}

// Returns the operations that read or write `v`, for example, `1, 29`. For a
// spilled vreg, these are the operations that load its stack slot or store to
// it.
std::string ops_str(const ir_t &ir, int v) {
    std::string s;
    std::vector<int> def_vregs, use_vregs;
    for (int i = 0; i < ir.n_ops(); i++) {
        ir.def_use(ir.ops()[i], def_vregs, use_vregs);
        if (std::find(def_vregs.begin(), def_vregs.end(), v) == def_vregs.end()
                && std::find(use_vregs.begin(), use_vregs.end(), v)
                        == use_vregs.end())
            continue;
        if (!s.empty()) s += ", ";
        s += std::to_string(i);
    }
    return s;
}

// Returns one line per spilled vreg, grouped by register file, with the
// operations that access its stack slot.
std::string spill_lines(const ir_t &ir, const kernel_info_t &info) {
    const reg_pools_t &pools = info.reg_cfg->pools;
    ostringstream_t ss;
    for (int f = 0; f < (int)pools.files.size(); f++)
        for (int v : file_vregs(ir, pools, f))
            if (info.alloc->assignments[v].spilled)
                ss << "spill " << vreg_str(ir, (vreg_t)v, &info) << ": ops "
                   << ops_str(ir, v) << "\n";
    return ss.str();
}

// Returns the IR dump, with the register allocation in `info` and the
// register pressure in `pressure` when they are not null (see `to_string()` in
// `dump.hpp`).
std::string ir_dump(const ir_t &ir, const kernel_info_t *info = nullptr,
        const std::vector<std::vector<int>> *pressure = nullptr) {
    ostringstream_t ss;

    // With the allocation, a header line names the columns, and the pressure
    // column has one number per register file.
    std::vector<std::string> names;
    if (info) {
        for (int f = 0; f < (int)info->reg_cfg->pools.files.size(); f++)
            names.push_back(file_name(info->reg_cfg->pools, f));
        ss << "index |";
        for (const std::string &name : names)
            ss << " " << name;
        ss << " | operation\n";
    }

    // Indentation follows the loop nesting. A loop's closing line is indented
    // like its opening line, so the body stands out between the two.
    int depth = 0;
    for (int i = 0; i < ir.n_ops(); i++) {
        const op_t &op = ir.ops()[i];
        if (op.kind == op_kind_t::loop_end) depth--;

        ss << std::setw(5) << i << " | ";
        if (info) {
            // Each number is right-aligned under the name of its file.
            for (int f = 0; f < (int)names.size(); f++)
                ss << std::setw((int)names[f].size()) << (*pressure)[f][i]
                   << " ";
            ss << "| ";
        }
        ss << std::string(2 * std::max(depth, 0), ' ')
           << op_str(ir, op, info, info ? &info->alloc->temps[i] : nullptr)
           << "\n";

        if (op.kind == op_kind_t::loop_begin) depth++;
    }
    return ss.str();
}

} // namespace

bool has_x64ir_token(const std::string &verbose_value) {
    // Tokens are split on `,` with no trimming, the same way the rest of
    // `ONEDNN_VERBOSE` is parsed.
    size_t pos = 0;
    while (true) {
        const size_t end = verbose_value.find(',', pos);
        if (verbose_value.compare(pos, end - pos, "x64ir") == 0) return true;
        if (end == std::string::npos) return false;
        pos = end + 1;
    }
}

std::string to_string(const ir_t &ir) {
    return ir_dump(ir);
}

std::string to_string(const ir_t &ir, const kernel_info_t &info) {
    assert(info.reg_cfg && info.alloc);
    const auto pressure = count_pressure(ir, info.reg_cfg->pools);
    return ir_dump(ir, &info, &pressure);
}

std::string format_kernel_dump(const kernel_info_t &info, const ir_t &ir) {
    assert(info.reg_cfg && info.alloc);
    const auto pressure = count_pressure(ir, info.reg_cfg->pools);
    ostringstream_t ss;
    ss << "begin x64ir " << info.name << " isa=" << isa2str(info.isa) << "\n";
    ss << "code: " << info.code_size << " bytes (instructions "
       << info.code_size - info.data_size << " bytes, static data "
       << info.data_size << " bytes)\n";
    ss << alloc_lines(ir, info, pressure);
    ss << spill_lines(ir, info);
    ss << "\n" << ir_dump(ir, &info, &pressure);
    ss << "end x64ir\n";
    return ss.str();
}

void print_kernel_dump(const jit_generator_t &gen, const ir_t &ir,
        const data_section_t &data, const reg_config_t &reg_cfg,
        const reg_alloc_result_t &alloc) {
    if (!is_dev_mode()) return;
    // `getenv_string_user()` lowercases the value, so `X64IR` works too.
    static const bool enabled = has_x64ir_token(getenv_string_user("VERBOSE"));
    if (!enabled) return;

    kernel_info_t info;
    info.name = gen.name();
    info.isa = gen.max_cpu_isa();
    info.code_size = gen.getSize();
    info.data_size = info.code_size - data.begin_offset;
    info.reg_cfg = &reg_cfg;
    info.alloc = &alloc;

    const std::string s = format_kernel_dump(info, ir);
    // One stdio call per kernel. POSIX locks the stream for the call, so the
    // output of kernels created at the same time does not mix.
    printf("%s", s.c_str());
    fflush(stdout);
}

} // namespace ir
} // namespace x64
} // namespace cpu
} // namespace impl
} // namespace dnnl
