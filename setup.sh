#!/bin/bash

# ─────────────────────────────────────────────────────────────────────────────
#  RAG Demo — Automated Setup Script
#  Arabic/French Document Q&A System
#
#  Usage:
#    chmod +x setup.sh
#    ./setup.sh
#
#  After setup, run the app with:
#    source venv/bin/activate
#    export OLLAMA_DEBUG=false
#    python3 rag.py 2>/dev/null     ← clean output (recommended)
#    python3 rag.py                 ← verbose output (for debugging)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ─── COLORS ──────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m'

print_step()    { echo -e "\n${BLUE}${BOLD}▶ $1${NC}"; }
print_success() { echo -e "${GREEN}✓ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠ $1${NC}"; }
print_error()   { echo -e "${RED}✗ $1${NC}"; exit 1; }

# ─── HEADER ──────────────────────────────────────────────────────────────────
echo -e "\n${BOLD}============================================================${NC}"
echo -e "${BOLD}   RAG Demo — Automated Setup${NC}"
echo -e "${BOLD}   Arabic/French Document Q&A System${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "${YELLOW}Note: This system runs on CPU. Each answer takes 1-2 minutes.${NC}"
echo -e "${YELLOW}      This is normal without a GPU.${NC}"

# ─── CHECK PYTHON ────────────────────────────────────────────────────────────
print_step "Checking Python version..."

if ! command -v python3 &> /dev/null; then
    print_error "Python3 not found. Please install Python 3.10 or higher."
fi

PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info.minor)')
PYTHON_VERSION="$PYTHON_MAJOR.$PYTHON_MINOR"

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    print_error "Python 3.10+ required. Found: $PYTHON_VERSION"
fi

print_success "Python $PYTHON_VERSION found"

# ─── CHECK / INSTALL OLLAMA ──────────────────────────────────────────────────
print_step "Checking Ollama..."

if ! command -v ollama &> /dev/null; then
    # This install path is intentionally Linux-oriented.
    # On macOS or other environments, install Ollama using the platform package.
    print_warning "Ollama not found. Installing..."
    curl -fsSL https://ollama.com/install.sh | sh
    print_success "Ollama installed"
else
    print_success "Ollama already installed ($(ollama --version))"
fi

# ─── START OLLAMA ────────────────────────────────────────────────────────────
print_step "Starting Ollama server..."

if curl -s http://localhost:11434 > /dev/null 2>&1; then
    print_success "Ollama already running at localhost:11434"
else
    OLLAMA_DEBUG=false ollama serve > /dev/null 2>&1 &
    echo -e "   Starting server..."
    # Wait for the local HTTP endpoint instead of assuming a fixed startup time.
    for _ in {1..30}; do
        if curl -s http://localhost:11434 > /dev/null 2>&1; then
            break
        fi
        sleep 1
    done

    if curl -s http://localhost:11434 > /dev/null 2>&1; then
        print_success "Ollama server started"
    else
        print_error "Ollama server failed to start. Try running 'ollama serve' manually in another terminal."
    fi
fi

# ─── PULL MODEL ──────────────────────────────────────────────────────────────
print_step "Checking Llama 3.2 3B model..."

if ollama list 2>/dev/null | grep -q "llama3.2:3b"; then
    print_success "llama3.2:3b already downloaded"
else
    print_warning "Downloading llama3.2:3b (~2GB) — this will take several minutes..."
    ollama pull llama3.2:3b
    print_success "llama3.2:3b downloaded"
fi

# ─── VIRTUAL ENVIRONMENT ─────────────────────────────────────────────────────
print_step "Setting up Python virtual environment..."

if [ -d "venv" ]; then
    print_warning "venv already exists — skipping creation"
else
    if ! python3 -m venv venv 2>/dev/null; then
        # This is the only distro-specific fallback in the script.
        # It keeps the setup simple for Ubuntu/Debian users.
        print_warning "python3-venv not found. Installing..."
        sudo apt-get install -y python3-venv > /dev/null 2>&1
        python3 -m venv venv
    fi
    print_success "Virtual environment created"
fi

source venv/bin/activate
print_success "Virtual environment activated"

# ─── INSTALL DEPENDENCIES ────────────────────────────────────────────────────
print_step "Installing Python dependencies..."
print_warning "First install takes 5-10 minutes — downloading PyTorch and other large packages."
echo ""

python3 -m pip install --upgrade pip --quiet

if [ -f "requirements.txt" ]; then
    python3 -m pip install -r requirements.txt --quiet
    print_success "Dependencies installed from requirements.txt"
else
    # Fallback only if requirements.txt is missing.
    python3 -m pip install \
        "langchain>=1.3.0" \
        "langchain-classic>=1.0.0" \
        "langchain-chroma>=1.1.0" \
        "langchain-huggingface>=1.2.0" \
        "langchain-ollama>=1.1.0" \
        "langchain-text-splitters>=1.0.0" \
        "chromadb>=1.0.0" \
        "sentence-transformers>=5.0.0" \
        --quiet
    print_success "Dependencies installed"
fi

# ─── VERIFY ──────────────────────────────────────────────────────────────────
print_step "Verifying installation..."

python3 -c "
packages = [
    ('langchain_text_splitters', 'langchain-text-splitters'),
    ('langchain_chroma',         'langchain-chroma'),
    ('langchain_huggingface',    'langchain-huggingface'),
    ('langchain_ollama',         'langchain-ollama'),
    ('langchain_classic',        'langchain-classic'),
    ('chromadb',                 'chromadb'),
    ('sentence_transformers',    'sentence-transformers'),
]
all_good = True
for module, name in packages:
    try:
        __import__(module)
        print(f'  ✓ {name}')
    except ImportError:
        print(f'  ✗ {name} — MISSING')
        all_good = False
import sys
sys.exit(0 if all_good else 1)
"

print_success "All packages verified"

# ─── DONE ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${GREEN}${BOLD}   Setup complete.${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "Run the app:"
echo ""
echo -e "   ${BOLD}source venv/bin/activate${NC}"
echo -e "   ${BOLD}export OLLAMA_DEBUG=false${NC}"
echo -e "   ${BOLD}python3 rag.py 2>/dev/null${NC}   ${YELLOW}← clean output (recommended)${NC}"
echo ""
echo -e "   or for debugging:"
echo -e "   ${BOLD}python3 rag.py${NC}               ${YELLOW}← verbose output with model logs${NC}"
echo ""
echo -e "${YELLOW}⚠ First question takes 30-60s to load the model.${NC}"
echo -e "${YELLOW}  Each answer takes 1-2 minutes on CPU. This is normal.${NC}"
echo ""
