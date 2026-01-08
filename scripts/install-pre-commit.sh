#!/bin/bash
#
# VecBoost Pre-commit Hook Installation Script
# =============================================
# This script installs and configures pre-commit hooks for the VecBoost project.
#
# Usage:
#   ./install-pre-commit.sh [--install]    # Install hooks
#   ./install-pre-commit.sh [--uninstall]  # Remove hooks
#   ./install-pre-commit.sh [--run]        # Run checks manually
#   ./install-pre-commit.sh [--help]       # Show this help
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

show_help() {
    echo ""
    echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║       VecBoost Pre-commit Hook Installation Script         ║${NC}"
    echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install     Install pre-commit hooks"
    echo "  --uninstall   Remove pre-commit hooks"
    echo "  --run         Run pre-commit checks manually"
    echo "  --help        Show this help message"
    echo ""
    echo "Installation Requirements:"
    echo "  - Python 3.7+ (for pre-commit)"
    echo "  - Rust toolchain (cargo, clippy)"
    echo ""
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."

    local missing=()

    # Check Python
    if ! command -v python3 &> /dev/null; then
        if ! command -v python &> /dev/null; then
            missing+=("Python 3")
        fi
    fi

    # Check Rust
    if ! command -v cargo &> /dev/null; then
        missing+=("Rust/Cargo")
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        missing+=("Git")
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        print_error "Missing prerequisites: ${missing[*]}"
        echo "Please install the missing tools and try again."
        exit 1
    fi

    print_success "All prerequisites met"
}

# Install pre-commit
install_hooks() {
    print_info "Installing pre-commit hooks..."

    check_prerequisites

    # Install or update pre-commit Python package
    print_info "Ensuring pre-commit is installed..."
    python3 -m pip install --upgrade pre-commit 2>/dev/null || \
        python3 -m pip install pre-commit 2>/dev/null || \
        pip install pre-commit 2>/dev/null || {
            print_warning "Could not install pre-commit via pip"
            print_info "Installing via pip3..."
            pip3 install pre-commit
        }

    # Install pre-commit hooks for this repository
    print_info "Installing Git hooks..."
    pre-commit install

    # Also install commit-msg hook if available
    if pre-commit install --hook-type commit-msg 2>/dev/null; then
        print_success "Commit-msg hook installed"
    fi

    # Update pre-commit hooks to latest versions
    print_info "Updating hooks to latest versions..."
    pre-commit autoupdate

    print_success "Pre-commit hooks installed successfully!"
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  Installation Complete!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "The following checks will now run automatically before each commit:"
    echo "  1. cargo fmt - Code formatting check"
    echo "  2. cargo clippy - Code quality check (strict)"
    echo "  3. cargo check - Compilation check"
    echo ""
    echo "To run checks manually:"
    echo -e "  ${YELLOW}$0 --run${NC}"
    echo ""
    echo "To skip hooks on a specific commit:"
    echo -e "  ${YELLOW}git commit --no-verify -m \"message\"${NC}"
    echo ""
}

# Uninstall pre-commit
uninstall_hooks() {
    print_info "Uninstalling pre-commit hooks..."

    if command -v pre-commit &> /dev/null; then
        pre-commit clean
        pre-commit uninstall 2>/dev/null || true
        print_success "Pre-commit hooks removed"
    else
        print_warning "pre-commit not found, hooks may not be installed"
    fi
}

# Run checks manually
run_checks() {
    print_info "Running pre-commit checks..."
    echo ""

    cd "$PROJECT_ROOT"
    bash "$SCRIPT_DIR/pre-commit-check.sh"
}

# Main
main() {
    case "${1:-}" in
        --install|-i)
            install_hooks
            ;;
        --uninstall|-u)
            uninstall_hooks
            ;;
        --run|-r)
            run_checks
            ;;
        --help|-h|"")
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"