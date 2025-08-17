#!/bin/bash

# Claude Code Audit Agent Installation Script
# This script installs the audit agent and configures Claude Code to use it

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLAUDE_DIR="$HOME/.claude"
BACKUP_DIR="$CLAUDE_DIR/backups/$(date +%Y%m%d_%H%M%S)"
AUDIT_LOG_DIR="$CLAUDE_DIR/audit_logs"

echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}     Claude Code Audit Agent Installation Script${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""

# Function to print status messages
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check if running from correct directory
if [[ ! -f "$SCRIPT_DIR/audit_agent.sh" ]]; then
    print_error "Required files not found. Please run this script from the Agent-Tools/Audit-Agent directory."
    exit 1
fi

# Step 1: Create necessary directories
echo -e "\n${BLUE}Step 1: Creating directories...${NC}"
mkdir -p "$CLAUDE_DIR"
mkdir -p "$AUDIT_LOG_DIR"
mkdir -p "$BACKUP_DIR"
print_status "Created directories"

# Step 2: Backup existing configuration
echo -e "\n${BLUE}Step 2: Backing up existing configuration...${NC}"
if [[ -f "$CLAUDE_DIR/settings.json" ]]; then
    cp "$CLAUDE_DIR/settings.json" "$BACKUP_DIR/settings.json.bak"
    print_status "Backed up existing settings.json to $BACKUP_DIR/"
else
    print_info "No existing settings.json found, creating new one"
fi

if [[ -f "$CLAUDE_DIR/audit_agent.sh" ]]; then
    cp "$CLAUDE_DIR/audit_agent.sh" "$BACKUP_DIR/audit_agent.sh.bak"
    print_status "Backed up existing audit_agent.sh"
fi

# Step 3: Install audit agent script
echo -e "\n${BLUE}Step 3: Installing audit agent...${NC}"
cp "$SCRIPT_DIR/audit_agent.sh" "$CLAUDE_DIR/audit_agent.sh"
chmod +x "$CLAUDE_DIR/audit_agent.sh"
print_status "Installed audit_agent.sh"

# Step 4: Install or merge global settings
echo -e "\n${BLUE}Step 4: Configuring Claude Code settings...${NC}"

if [[ -f "$CLAUDE_DIR/settings.json" ]]; then
    # Check if Python is available for JSON merging
    if command -v python3 &> /dev/null; then
        print_info "Merging with existing settings.json..."
        
        # Create a Python script to merge JSON
        cat > /tmp/merge_settings.py << 'EOF'
import json
import sys

def merge_hooks(existing, new):
    """Merge hooks from new settings into existing settings."""
    if "hooks" not in existing:
        existing["hooks"] = {}
    
    for hook_type, hook_list in new.get("hooks", {}).items():
        if hook_type not in existing["hooks"]:
            existing["hooks"][hook_type] = []
        
        # Check if similar hooks already exist
        for new_hook in hook_list:
            hook_exists = False
            for existing_hook in existing["hooks"][hook_type]:
                if existing_hook.get("matcher") == new_hook.get("matcher"):
                    hook_exists = True
                    break
            
            if not hook_exists:
                existing["hooks"][hook_type].append(new_hook)
    
    # Add audit configuration if not present
    if "audit" not in existing:
        existing["audit"] = new.get("audit", {})
    
    # Add tools configuration if not present
    if "tools" not in existing:
        existing["tools"] = new.get("tools", {})
    
    return existing

# Read existing settings
with open(sys.argv[1], 'r') as f:
    existing = json.load(f)

# Read new settings
with open(sys.argv[2], 'r') as f:
    new = json.load(f)

# Merge settings
merged = merge_hooks(existing, new)

# Write merged settings
with open(sys.argv[1], 'w') as f:
    json.dump(merged, f, indent=2)

print("Settings merged successfully")
EOF
        
        python3 /tmp/merge_settings.py "$CLAUDE_DIR/settings.json" "$SCRIPT_DIR/global_settings.json"
        rm /tmp/merge_settings.py
        print_status "Merged audit hooks into existing settings.json"
    else
        print_warning "Python not found. Please manually merge the hooks from global_settings.json"
        print_info "You can view the required hooks in: $SCRIPT_DIR/global_settings.json"
    fi
else
    # No existing settings, just copy the new one
    cp "$SCRIPT_DIR/global_settings.json" "$CLAUDE_DIR/settings.json"
    print_status "Created new settings.json with audit hooks"
fi

# Step 5: Install project configuration
echo -e "\n${BLUE}Step 5: Setting up project configuration...${NC}"
PROJECT_CLAUDE_DIR=".claude"

if [[ -d "$PROJECT_CLAUDE_DIR" ]]; then
    if [[ ! -f "$PROJECT_CLAUDE_DIR/settings.json" ]]; then
        cp "$SCRIPT_DIR/project_settings.json" "$PROJECT_CLAUDE_DIR/settings.json"
        print_status "Installed project settings.json"
    else
        print_warning "Project settings.json already exists. Skipping to avoid overwriting."
        print_info "You can manually merge from: $SCRIPT_DIR/project_settings.json"
    fi
    
    if [[ ! -f "$PROJECT_CLAUDE_DIR/audit_config.json" ]]; then
        cp "$SCRIPT_DIR/audit_config.json" "$PROJECT_CLAUDE_DIR/audit_config.json"
        print_status "Installed audit_config.json"
    else
        print_warning "audit_config.json already exists in project"
    fi
else
    print_info "No .claude directory in current project. Skipping project configuration."
    print_info "To add project-specific configuration, create a .claude directory and copy the files."
fi

# Step 6: Verify installation
echo -e "\n${BLUE}Step 6: Verifying installation...${NC}"

# Check if audit script is executable
if [[ -x "$CLAUDE_DIR/audit_agent.sh" ]]; then
    print_status "Audit agent is executable"
else
    print_error "Audit agent is not executable"
    chmod +x "$CLAUDE_DIR/audit_agent.sh"
    print_status "Fixed permissions"
fi

# Test the audit script
echo -e "\n${BLUE}Testing audit agent...${NC}"
TEST_FILE="/tmp/test_audit_$(date +%s).py"
echo "# Test file for audit" > "$TEST_FILE"
echo "print('test')" >> "$TEST_FILE"

if "$CLAUDE_DIR/audit_agent.sh" "Test" "$TEST_FILE" &> /dev/null; then
    print_status "Audit agent test successful"
    rm "$TEST_FILE"
else
    print_warning "Audit agent test failed. Please check the installation."
fi

# Check if logs are being created
if [[ -d "$AUDIT_LOG_DIR" ]]; then
    print_status "Audit log directory exists"
else
    print_error "Audit log directory not found"
fi

# Display summary
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}     Installation Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BLUE}Installed components:${NC}"
echo "  • Audit agent: $CLAUDE_DIR/audit_agent.sh"
echo "  • Settings: $CLAUDE_DIR/settings.json"
echo "  • Audit logs: $AUDIT_LOG_DIR/"
if [[ -d "$BACKUP_DIR" ]] && [[ "$(ls -A $BACKUP_DIR)" ]]; then
    echo "  • Backups: $BACKUP_DIR/"
fi
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. Restart Claude Code for settings to take effect"
echo "  2. The audit agent will now run automatically after file modifications"
echo "  3. Check audit logs at: $AUDIT_LOG_DIR/"
echo "  4. Customize audit rules in: .claude/audit_config.json"
echo ""
echo -e "${YELLOW}Note:${NC} If you experience any issues, you can restore your backup from:"
echo "  $BACKUP_DIR/"
echo ""
print_info "For more information, see README.md"