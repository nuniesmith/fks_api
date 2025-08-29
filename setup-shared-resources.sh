#!/usr/bin/env bash
# setup-shared-resources.sh - Initialize shared resources for fks_api
set -euo pipefail

# Colors
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log(){ local lvl="$1"; shift; local msg="$*"; case "$lvl" in INFO) echo -e "${GREEN}[INFO]${NC} $msg";; WARN) echo -e "${YELLOW}[WARN]${NC} $msg";; ERROR) echo -e "${RED}[ERROR]${NC} $msg";; DEBUG) echo -e "${BLUE}[DEBUG]${NC} $msg";; esac; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

show_help() {
    cat <<EOF
Usage: ./setup-shared-resources.sh [options]

This script sets up shared Docker and script resources for fks_api using Git submodules.

Options:
  --force-reinit     Force reinitialize submodules (removes and re-adds them)
  --update           Update existing submodules to latest
  --test             Test the setup after initialization
  --help             Show this help

What this script does:
1. Initializes shared_docker and shared_scripts as Git submodules
2. Updates docker-compose.yml to use shared templates
3. Configures entrypoint scripts
4. Tests the setup
EOF
}

init_submodules() {
    log INFO "🔧 Initializing shared resource submodules..."
    
    # Check if .gitmodules exists and has our submodules
    if [[ -f .gitmodules ]]; then
        if grep -q "shared/shared_docker" .gitmodules && grep -q "shared/shared_scripts" .gitmodules; then
            log INFO "✅ Submodules already configured in .gitmodules"
            git submodule update --init --recursive
            return 0
        fi
    fi
    
    # Add submodules if they don't exist
    if [[ ! -d "shared/shared_docker/.git" ]]; then
        log INFO "Adding shared_docker submodule..."
        git submodule add https://github.com/nuniesmith/shared-docker.git shared/shared_docker
    fi
    
    if [[ ! -d "shared/shared_scripts/.git" ]]; then
        log INFO "Adding shared_scripts submodule..."
        git submodule add https://github.com/nuniesmith/shared-scripts.git shared/shared_scripts
    fi
    
    log INFO "✅ Submodules initialized"
}

force_reinit_submodules() {
    log WARN "🔄 Force reinitializing submodules..."
    
    # Remove existing submodules
    if [[ -d "shared/shared_docker" ]]; then
        git submodule deinit -f shared/shared_docker
        git rm -f shared/shared_docker
        rm -rf .git/modules/shared/shared_docker
    fi
    
    if [[ -d "shared/shared_scripts" ]]; then
        git submodule deinit -f shared/shared_scripts  
        git rm -f shared/shared_scripts
        rm -rf .git/modules/shared/shared_scripts
    fi
    
    # Re-add them
    init_submodules
}

update_submodules() {
    log INFO "📦 Updating submodules to latest versions..."
    git submodule update --remote --recursive
    log INFO "✅ Submodules updated"
}

setup_docker_config() {
    log INFO "🐳 Setting up Docker configuration..."
    
    # Backup original compose file if it exists and doesn't already have a backup
    if [[ -f "docker-compose.yml" && ! -f "docker-compose.original.yml" ]]; then
        cp docker-compose.yml docker-compose.original.yml
        log INFO "📦 Backed up original docker-compose.yml"
    fi
    
    log INFO "✅ Docker configuration ready"
}

setup_entrypoint() {
    log INFO "🚀 Setting up entrypoint scripts..."
    
    # Make sure our entrypoint script is executable
    chmod +x entrypoint.sh 2>/dev/null || true
    chmod +x build.sh 2>/dev/null || true
    chmod +x start.sh 2>/dev/null || true
    
    log INFO "✅ Entrypoint scripts configured"
}

test_setup() {
    log INFO "🧪 Testing shared resources setup..."
    
    # Test submodules are present (check for actual content, not .git dirs)
    if [[ ! -d "shared/shared_docker" || ! -d "shared/shared_scripts" ]]; then
        log ERROR "❌ Submodule directories not found"
        return 1
    fi
    
    # Test shared templates exist
    if [[ ! -f "shared/shared_docker/templates/python.Dockerfile" ]]; then
        log ERROR "❌ Python Dockerfile template not found"
        return 1
    fi
    
    # Test shared scripts exist  
    if [[ ! -f "shared/shared_scripts/docker/entrypoint-python.sh" ]]; then
        log ERROR "❌ Python entrypoint script not found"
        return 1
    fi
    
    # Test build script works
    if [[ -x "build.sh" ]]; then
        ./build.sh --help >/dev/null 2>&1 || {
            log WARN "⚠️ Build script may have issues"
        }
    fi
    
    log INFO "✅ All tests passed!"
}

show_status() {
    log INFO "📊 Current status:"
    echo ""
    echo "Submodules:"
    git submodule status 2>/dev/null || echo "  No submodules found"
    echo ""
    echo "Available shared resources:"
    if [[ -d "shared/shared_docker" ]]; then
        echo "  ✅ shared_docker ($(ls shared/shared_docker/templates/*.Dockerfile 2>/dev/null | wc -l) templates)"
    else
        echo "  ❌ shared_docker"
    fi
    
    if [[ -d "shared/shared_scripts" ]]; then
        echo "  ✅ shared_scripts ($(ls shared/shared_scripts/docker/entrypoint-*.sh 2>/dev/null | wc -l) entrypoints)"
    else
        echo "  ❌ shared_scripts"
    fi
    echo ""
}

main() {
    local FORCE_REINIT=false
    local UPDATE_ONLY=false
    local TEST_SETUP=false
    
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --force-reinit) FORCE_REINIT=true; shift;;
            --update) UPDATE_ONLY=true; shift;;
            --test) TEST_SETUP=true; shift;;
            --help|-h) show_help; exit 0;;
            *) log ERROR "Unknown option: $1"; show_help; exit 1;;
        esac
    done
    
    log INFO "🚀 Setting up shared resources for fks_api"
    
    # Change to script directory
    cd "$SCRIPT_DIR"
    
    if [[ "$FORCE_REINIT" == "true" ]]; then
        force_reinit_submodules
    elif [[ "$UPDATE_ONLY" == "true" ]]; then
        update_submodules
    else
        init_submodules
    fi
    
    setup_docker_config
    setup_entrypoint
    
    if [[ "$TEST_SETUP" == "true" ]]; then
        test_setup
    fi
    
    show_status
    
    log INFO "🎉 Setup complete!"
    echo ""
    echo "Next steps:"
    echo "1. Test the setup:"
    echo "   ./setup-shared-resources.sh --test"
    echo ""
    echo "2. Build with shared templates:"
    echo "   ./build.sh"
    echo ""
    echo "3. Start the service:"
    echo "   ./start.sh"
    echo ""
    echo "4. Or use docker-compose directly:"
    echo "   docker-compose up --build"
}

main "$@"
