#!/usr/bin/env bash
# =============================================================================
# build-oneapi-cuda-deb.sh
#
# Сборка ik_llama.cpp с Intel oneAPI (icpx) + CUDA + AVX-512 + MOE
# → динамически слинкованный deb-пакет для Ubuntu (amd64)
#
# Подход: BUILD_SHARED_LIBS=ON — бинарники 4-12MB, вся тяжесть в .so.
# nvcc stubs живут в libggml.so, не в бинарнике → нет PC32 overflow.
# Динамические зависимости (CUDA, NCCL, Intel runtime) объявлены в deb.
#
# Требования:
#   - Intel oneAPI DPC++/C++ Compiler (icpx, icx, lld)
#   - NVIDIA CUDA Toolkit (nvcc, libcudart12, libcublas12)
#   - libnccl-dev (NCCL headers)
#   - CMake >= 3.18, Ninja, dpkg-deb
#
# Ограничения:
#   Одна CUDA архитектура (sm_86 = RTX 3090).
#   2 archs → libggml.so ~2GB → ALL_QUANTS ON вызывает PC32 overflow.
#   С 1 arch → libggml.so ~1GB → все ALL_QUANTS ON помещаются.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="${1:-build-oneapi}"
DEB_STAGING="${BUILD_DIR}/deb-staging"
JOBS="$(nproc)"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
die()   { error "$@"; exit 1; }

get_version() {
    cd "$PROJECT_DIR"
    local ver
    if ver="$(git describe --tags --always 2>/dev/null)"; then
        ver="${ver#v}"
        ver="${ver%%-*}"
        echo "$ver"
    elif ver="$(git rev-parse --short HEAD 2>/dev/null)"; then
        echo "0.0.${ver}"
    else
        echo "1.0.0"
    fi
}

setup_oneapi() {
    if ! command -v icpx &>/dev/null; then
        if [[ -f /opt/intel/oneapi/setvars.sh ]]; then
            info "oneAPI не в PATH, source /opt/intel/oneapi/setvars.sh"
            set +u
            source /opt/intel/oneapi/setvars.sh 2>/dev/null || true
            set -u
        fi
    fi

    local LLD_DIR=""
    for candidate in \
        "/opt/intel/oneapi/compiler/2026.1/bin/compiler" \
        "/opt/intel/oneapi/compiler/latest/bin/compiler"; do
        if [[ -x "${candidate}/ld.lld" ]]; then
            LLD_DIR="$candidate"
            break
        fi
    done

    if [[ -n "$LLD_DIR" ]]; then
        mkdir -p /tmp/lld-bin
        ln -sf "${LLD_DIR}/ld.lld" /tmp/lld-bin/ld.lld 2>/dev/null || true
        ln -sf "${LLD_DIR}/lld" /tmp/lld-bin/lld 2>/dev/null || true
        export PATH="/tmp/lld-bin:${PATH}"
        ok "lld добавлен в PATH"
    else
        die "Intel lld не найден"
    fi
}

check_deps() {
    info "Проверка зависимостей..."
    local missing=()

    for cmd in icpx icx nvcc cmake ninja dpkg-deb; do
        if ! command -v "$cmd" &>/dev/null; then
            case "$cmd" in
                icpx)     missing+=("Intel oneAPI DPC++/C++ Compiler (icpx)") ;;
                icx)      missing+=("Intel oneAPI C Compiler (icx)") ;;
                nvcc)     missing+=("NVIDIA CUDA Toolkit (nvcc)") ;;
                cmake)    missing+=("cmake (>= 3.18)") ;;
                ninja)    missing+=("ninja (пакет: ninja-build)") ;;
                dpkg-deb) missing+=("dpkg-deb (пакет: dpkg-dev)") ;;
            esac
        fi
    done

    # NCCL headers
    if [[ ! -f /usr/include/nccl.h ]]; then
        missing+=("libnccl-dev (NCCL headers)")
    fi

    if [[ ${#missing[@]} -gt 0 ]]; then
        error "Отсутствуют зависимости:"
        for dep in "${missing[@]}"; do
            echo "  - $dep"
        done
        exit 1
    fi

    ok "Все зависимости найдены"
}

print_versions() {
    info "Версии инструментов:"
    echo "  icpx:    $(icpx --version 2>&1 | head -1)"
    echo "  nvcc:    $(nvcc --version 2>&1 | tail -1)"
    echo "  lld:     $(lld --version 2>&1 | head -1)"
    echo "  cmake:   $(cmake --version 2>&1 | head -1)"
    echo "  ninja:   $(ninja --version 2>&1)"
    echo ""
}

cmake_configure() {
    info "Конфигурация CMake (shared libs, CUDA sm_86, ALL_QUANTS ON)..."

    cd "$PROJECT_DIR"

    local cmake_args=(
        -G Ninja
        -B "$BUILD_DIR"
        -DCMAKE_BUILD_TYPE=Release

        # Компиляторы
        -DCMAKE_C_COMPILER=icx
        -DCMAKE_CXX_COMPILER=icpx

        # PIC + mcmodel=large (чтобы icpx/icx не генерировал PC32)
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DCMAKE_C_FLAGS=-mcmodel=large
        -DCMAKE_CXX_FLAGS=-mcmodel=large

        # CPU: AVX-512 максимум + native для build-машины
        -DGGML_NATIVE=ON
        -DGGML_AVX512=ON
        -DGGML_AVX512_VBMI=ON
        -DGGML_AVX512_VNNI=ON
        -DGGML_AVX512_BF16=ON

        # Динамические библиотеки — бинарники маленькие
        -DBUILD_SHARED_LIBS=ON

        # LTO off (icpx LTO → LLVM bitcode)
        -DGGML_LTO=OFF

        # CUDA: sm_86 (RTX 3090) — одна архитектура,
        # чтобы libggml.so был ~1GB и все ALL_QUANTS поместились
        -DGGML_CUDA=ON
        -DCMAKE_CUDA_ARCHITECTURES="86"

        # CUDA: ВСЕ кванты включены (116 cross-type fattn-vec ядер)
        -DGGML_CUDA_FA_ALL_QUANTS=ON
        -DGGML_CUDA_USE_GRAPHS=ON
        -DGGML_CUDA_FUSION=1

        # IQK: ВСЕ кванты включены (CPU FA: Q4_0/Q4_1/IQ4_NL как K-type)
        -DGGML_IQK_MUL_MAT=ON
        -DGGML_IQK_FLASH_ATTENTION=ON
        -DGGML_IQK_FA_ALL_QUANTS=ON

        # OpenMP, NCCL
        -DGGML_OPENMP=ON
        -DGGML_NCCL=ON

        # lld как линкер
        -DCMAKE_EXE_LINKER_FLAGS="-fuse-ld=lld -Wl,--allow-shlib-undefined"
        -DCMAKE_SHARED_LINKER_FLAGS="-fuse-ld=lld -Wl,-z,noseparate-code"

        # GNU ar/ranlib
        -DCMAKE_AR=/usr/bin/ar
        -DCMAKE_RANLIB=/usr/bin/ranlib

        # libcurl
        -DLLAMA_CURL=ON

        # Сборка
        -DGGML_BUILD_TESTS=OFF
        -DGGML_BUILD_EXAMPLES=OFF
        -DLLAMA_BUILD_TESTS=OFF
        -DLLAMA_BUILD_EXAMPLES=ON
        -DLLAMA_BUILD_SERVER=ON
    )

    cmake "${cmake_args[@]}"
    ok "Конфигурация завершена"
}

cmake_build() {
    info "Сборка (jobs=${JOBS})..."
    cd "$PROJECT_DIR"
    cmake --build "$BUILD_DIR" --config Release -j"$JOBS"
    ok "Сборка завершена"
}

verify_binaries() {
    info "Проверка собранных бинарников..."

    local bin_dir="${BUILD_DIR}/bin"
    local required_bins=("llama-server" "llama-cli" "llama-bench" "llama-quantize")
    local all_found=true

    for bin in "${required_bins[@]}"; do
        local bin_path="${bin_dir}/${bin}"
        if [[ -f "$bin_path" ]]; then
            local size
            size="$(du -h "$bin_path" | cut -f1)"
            ok "  ${bin} (${size})"
        else
            error "  ${bin} — НЕ НАЙДЕН"
            all_found=false
        fi
    done

    if [[ "$all_found" != "true" ]]; then
        die "Не все бинарники собраны"
    fi

    # Показать .so
    info "Собранные shared libraries:"
    find "$BUILD_DIR" -name "libggml*.so" -o -name "libllama*.so" -o -name "libmtmd*.so" 2>/dev/null | while read f; do
        echo "  $(du -h "$f" | cut -f1) $f"
    done
    echo ""

    # Проверка запуска (в build-дереве, с ld_LIBRARY_PATH для Intel runtime)
    local INTEL_LIB=""
    for candidate in /opt/intel/oneapi/compiler/*/lib; do
        [[ -d "$candidate" ]] && INTEL_LIB="$candidate" && break
    done

    info "Проверка запуска llama-cli:"
    LD_LIBRARY_PATH="${BUILD_DIR}/ggml/src:${BUILD_DIR}/src:${BUILD_DIR}/examples/mtmd${INTEL_LIB:+:$INTEL_LIB}" \
        "${bin_dir}/llama-cli" --version 2>&1 | head -3 || true
    echo ""
}

# ─── Сборка deb-пакета ──────────────────────────────────────────────────────
build_deb() {
    local version
    version="$(get_version)"
    local pkg_name="ik-llama-cuda"
    local pkg_version="${version}"
    local pkg_arch="amd64"
    local pkg_dir="${DEB_STAGING}/${pkg_name}_${pkg_version}_${pkg_arch}"

    info "Сборка deb-пакета: ${pkg_name}_${pkg_version}_${pkg_arch}.deb"

    rm -rf "$DEB_STAGING"
    mkdir -p "$pkg_dir/DEBIAN"
    mkdir -p "$pkg_dir/usr/bin"
    mkdir -p "$pkg_dir/usr/lib/ik-llama"
    mkdir -p "$pkg_dir/etc/ld.so.conf.d"

    # ─── Бинарники ────────────────────────────────────────────────────────
    local bin_dir="${BUILD_DIR}/bin"
    local bins_to_package=()
    for bin_path in "${bin_dir}"/llama-*; do
        [[ -f "$bin_path" ]] || continue
        local bin_name
        bin_name="$(basename "$bin_path")"
        # skip symlinks and wrapper scripts
        [[ -L "$bin_path" ]] && continue
        bins_to_package+=("$bin_name")
    done

    info "Найдено бинарников для пакета: ${#bins_to_package[@]}"
    for bin in "${bins_to_package[@]}"; do
        cp "${bin_dir}/${bin}" "$pkg_dir/usr/bin/"
        chmod 755 "$pkg_dir/usr/bin/${bin}"
    done

    # ─── Shared libraries ─────────────────────────────────────────────────
    # ggml .so files
    for so in "${BUILD_DIR}"/ggml/src/libggml*.so; do
        [[ -f "$so" ]] || continue
        cp "$so" "$pkg_dir/usr/lib/ik-llama/"
    done

    # llama .so
    for so in "${BUILD_DIR}"/src/libllama*.so; do
        [[ -f "$so" ]] || continue
        cp "$so" "$pkg_dir/usr/lib/ik-llama/"
    done

    # mtmd .so
    for so in "${BUILD_DIR}"/examples/mtmd/libmtmd*.so; do
        [[ -f "$so" ]] || continue
        cp "$so" "$pkg_dir/usr/lib/ik-llama/"
    done

    # ─── Intel runtime .so (обязательны для работы) ───────────────────────
    local INTEL_LIB_DIR="/opt/intel/oneapi/compiler/2026.1/lib"
    local intel_libs=(libiomp5.so libsvml.so libirng.so libimf.so libintlc.so.5)
    for lib in "${intel_libs[@]}"; do
        if [[ -f "${INTEL_LIB_DIR}/${lib}" ]]; then
            cp "${INTEL_LIB_DIR}/${lib}" "$pkg_dir/usr/lib/ik-llama/"
            ok "  bundled ${lib}"
        else
            warn "  ${lib} не найден в ${INTEL_LIB_DIR}"
        fi
    done

    # ─── Wrapper-скрипты с LD_LIBRARY_PATH ───────────────────────────────
    for bin_name in "${bins_to_package[@]}"; do
        local bin_path="$pkg_dir/usr/bin/${bin_name}"
        if [[ -f "$bin_path" ]]; then
            mv "$bin_path" "${bin_path}.bin"
            cat > "$bin_path" << WRAPPER
#!/bin/bash
export LD_LIBRARY_PATH="/usr/lib/ik-llama:\${LD_LIBRARY_PATH:-}"
exec /usr/bin/${bin_name}.bin "\$@"
WRAPPER
            chmod 755 "$bin_path"
        fi
    done

    # ─── ldconfig conf ────────────────────────────────────────────────────
    echo "/usr/lib/ik-llama" > "$pkg_dir/etc/ld.so.conf.d/ik-llama.conf"

    # ─── DEBIAN/control ───────────────────────────────────────────────────
    # Динамические зависимости:
    #   libcudart12  — CUDA runtime (libcuda.so от NVIDIA драйвера)
    #   libcublas12  — cuBLAS
    #   libnccl2     — NCCL
    #   libcurl4     — HTTP (скачивание моделей)
    #   libstdc++6   — C++ runtime
    local deps="libc6 (>= 2.31), libstdc++6 (>= 12), libcurl4, libcudart12, libcublas12, libnccl2"

    cat > "$pkg_dir/DEBIAN/control" << EOF
Package: ${pkg_name}
Version: ${pkg_version}
Section: science
Priority: optional
Architecture: ${pkg_arch}
Depends: ${deps}
Installed-Size: $(du -sk "$pkg_dir" | cut -f1)
Maintainer: ik_llama.cpp build script <build@ik-llama.local>
Homepage: https://github.com/ikawrakow/ik_llama.cpp
Description: ik_llama.cpp with Intel oneAPI + CUDA + AVX-512 + MOE (shared libs, all quants)
 High-performance LLM inference engine built with Intel oneAPI DPC++
 compiler (icpx) and NVIDIA CUDA. Includes optimized AVX-512 kernels,
 IQK FlashAttention, MOE prefetch engine, and NCCL multi-GPU support.
 .
 ALL quants enabled: 116 cross-type FlashAttention CUDA kernels,
 all MMQ/MMVQ matrix multiply kernels for 40+ quant types,
 all IQK CPU kernels with AVX-512 VNNI/BF16 support.
 .
 Shared libraries bundled: libggml.so (1 arch: sm_86 RTX 3090),
 libllama.so, libmtmd.so, Intel runtime (libiomp5, libsvml, libirng,
 libimf, libintlc).
 .
 Runtime dependencies resolved via apt:
   - libcudart12, libcublas12 (CUDA)
   - libnccl2 (NCCL multi-GPU)
   - libcurl4 (HTTP model downloads)
 .
 CUDA compute capability: sm_86 (RTX 3090 / RTX 3080 Ti).
EOF

    # ─── postinst: ldconfig ───────────────────────────────────────────────
    cat > "$pkg_dir/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
ldconfig
POSTINST
    chmod 755 "$pkg_dir/DEBIAN/postinst"

    # ─── postrm: ldconfig ─────────────────────────────────────────────────
    cat > "$pkg_dir/DEBIAN/postrm" << 'POSTRM'
#!/bin/bash
ldconfig
POSTRM
    chmod 755 "$pkg_dir/DEBIAN/postrm"

    # ─── Сборка ───────────────────────────────────────────────────────────
    dpkg-deb --root-owner-group --build "$pkg_dir"

    local deb_path="${DEB_STAGING}/${pkg_name}_${pkg_version}_${pkg_arch}.deb"
    local deb_size
    deb_size="$(du -h "$deb_path" | cut -f1)"

    ok "deb-пакет создан: ${deb_path} (${deb_size})"

    cp "$deb_path" "$PROJECT_DIR/"
    ok "Скопирован в: ${PROJECT_DIR}/${pkg_name}_${pkg_version}_${pkg_arch}.deb"

    echo ""
    info "Установка:"
    echo "  sudo apt install ./ik-llama-cuda_${pkg_version}_${pkg_arch}.deb"
    echo ""
    info "Проверка:"
    echo "  llama-server --version"
    echo "  llama-cli --version"
}

cleanup_old_deb() {
    local pkg_name="ik-llama-cuda"
    local old_debs
    old_debs="$(find "$PROJECT_DIR" -maxdepth 1 -name "${pkg_name}_*_amd64.deb" 2>/dev/null)"
    if [[ -n "$old_debs" ]]; then
        info "Удаляю старые пакеты:"
        echo "$old_debs" | while read -r f; do
            echo "  $(basename "$f")"
            rm -f "$f"
        done
    fi
}

cleanup_build_dir() {
    info "Удаляю build-директорию: ${BUILD_DIR}"
    rm -rf "$BUILD_DIR"
    ok "Build-директория удалена"
}

cleanup_old_builds() {
    info "Удаляю старые build-директории..."
    local count=0
    for d in "${PROJECT_DIR}"/build*; do
        [[ -d "$d" ]] || continue
        [[ "$d" == "${PROJECT_DIR}/${BUILD_DIR}" ]] && continue
        rm -rf "$d"
        ok "  удалена $(basename "$d")"
        ((count++))
    done
    if [[ $count -eq 0 ]]; then
        info "  старых build-директорий нет"
    fi
}

tag_build() {
    local version
    version="$(get_version)"
    cd "$PROJECT_DIR"

    if git rev-parse "$version" &>/dev/null; then
        warn "Тег ${version} уже существует, пропускаю"
    else
        git tag -a "$version" -m "deb: ${version}" 2>/dev/null
        ok "Создан тег: ${version}"
    fi
}

main() {
    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo "  ik_llama.cpp — Intel oneAPI + CUDA + AVX-512 + MOE builder"
    echo "  (shared libs mode — multi-arch CUDA via apt dependencies)"
    echo "═══════════════════════════════════════════════════════════════════"
    echo ""

    setup_oneapi
    check_deps
    print_versions
    cleanup_old_deb
    cleanup_old_builds
    cmake_configure
    cmake_build
    verify_binaries
    build_deb
    tag_build
    cleanup_build_dir

    echo ""
    echo "═══════════════════════════════════════════════════════════════════"
    echo -e "  ${GREEN}Сборка завершена успешно!${NC}"
    echo "═══════════════════════════════════════════════════════════════════"
}

main "$@"
