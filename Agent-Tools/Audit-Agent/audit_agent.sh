#!/bin/bash

# Claude Code Audit Agent
# Runs after every file write operation to audit changes

# Configuration
AUDIT_LOG_DIR="$HOME/.claude/audit_logs"
AUDIT_CONFIG="$HOME/.claude/audit_config.json"
PROJECT_CONFIG=".claude/audit_config.json"
DATE_FORMAT="%Y-%m-%d %H:%M:%S"
LOG_FILE="$AUDIT_LOG_DIR/audit_$(date +%Y%m%d).log"

# Ensure audit log directory exists
mkdir -p "$AUDIT_LOG_DIR"

# Function to log messages
log_message() {
    local level="$1"
    local message="$2"
    echo "[$(date +"$DATE_FORMAT")] [$level] $message" >> "$LOG_FILE"
}

# Function to check for sensitive data patterns
check_sensitive_data() {
    local file_path="$1"
    local issues=""
    
    # Check for common sensitive patterns
    if grep -qE "(api[_-]?key|secret|password|token|credential)" "$file_path" 2>/dev/null; then
        issues="${issues}SENSITIVE_DATA "
    fi
    
    # Check for hardcoded credentials
    if grep -qE "(['\"])(AIza|sk-|ghp_|ghs_|pat_|github_pat_)" "$file_path" 2>/dev/null; then
        issues="${issues}HARDCODED_CREDENTIALS "
    fi
    
    echo "$issues"
}

# Function to check for code quality issues
check_code_quality() {
    local file_path="$1"
    local extension="${file_path##*.}"
    local issues=""
    
    case "$extension" in
        py)
            # Check for Python issues
            if grep -qE "^\s*print\(" "$file_path" 2>/dev/null; then
                issues="${issues}DEBUG_PRINT "
            fi
            if grep -qE "# TODO|# FIXME|# HACK" "$file_path" 2>/dev/null; then
                issues="${issues}TODO_COMMENT "
            fi
            ;;
        js|ts|jsx|tsx)
            # Check for JavaScript/TypeScript issues
            if grep -qE "console\.(log|error|warn|debug)" "$file_path" 2>/dev/null; then
                issues="${issues}CONSOLE_LOG "
            fi
            if grep -qE "// TODO|// FIXME|// HACK" "$file_path" 2>/dev/null; then
                issues="${issues}TODO_COMMENT "
            fi
            if grep -qE "debugger;" "$file_path" 2>/dev/null; then
                issues="${issues}DEBUGGER_STATEMENT "
            fi
            ;;
        sh|bash)
            # Check for shell script issues
            if grep -qE "set -x" "$file_path" 2>/dev/null; then
                issues="${issues}DEBUG_MODE "
            fi
            ;;
    esac
    
    echo "$issues"
}

# Function to run project-specific checks
run_project_checks() {
    local file_path="$1"
    local project_root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
    
    # Run linting if available
    if [[ -f "$project_root/package.json" ]]; then
        # Check if file is JavaScript/TypeScript
        if [[ "$file_path" =~ \.(js|ts|jsx|tsx)$ ]]; then
            if command -v npm >/dev/null 2>&1; then
                # Try to run eslint if available
                if npm list eslint >/dev/null 2>&1; then
                    cd "$project_root"
                    npm run lint -- "$file_path" 2>&1 | head -5 >> "$LOG_FILE"
                fi
            fi
        fi
    fi
    
    # Run Python checks if available
    if [[ "$file_path" =~ \.py$ ]]; then
        if command -v ruff >/dev/null 2>&1; then
            ruff check "$file_path" 2>&1 | head -5 >> "$LOG_FILE"
        elif command -v flake8 >/dev/null 2>&1; then
            flake8 "$file_path" 2>&1 | head -5 >> "$LOG_FILE"
        fi
    fi
}

# Main audit function
audit_file() {
    local tool_name="$1"
    local file_path="$2"
    local old_content="$3"
    local new_content="$4"
    
    log_message "INFO" "Audit triggered by $tool_name for file: $file_path"
    
    # Check if file exists
    if [[ ! -f "$file_path" ]]; then
        log_message "WARNING" "File does not exist: $file_path"
        return 1
    fi
    
    # Get file stats
    local file_size=$(stat -f%z "$file_path" 2>/dev/null || stat -c%s "$file_path" 2>/dev/null)
    local file_lines=$(wc -l < "$file_path")
    log_message "INFO" "File stats: Size=${file_size} bytes, Lines=${file_lines}"
    
    # Check for sensitive data
    local sensitive_issues=$(check_sensitive_data "$file_path")
    if [[ -n "$sensitive_issues" ]]; then
        log_message "WARNING" "Sensitive data detected: $sensitive_issues"
        echo "⚠️  WARNING: Potential sensitive data detected in $file_path: $sensitive_issues" >&2
    fi
    
    # Check code quality
    local quality_issues=$(check_code_quality "$file_path")
    if [[ -n "$quality_issues" ]]; then
        log_message "INFO" "Code quality issues: $quality_issues"
    fi
    
    # Run project-specific checks
    run_project_checks "$file_path"
    
    # Log git diff if in a git repository
    if git rev-parse --git-dir >/dev/null 2>&1; then
        local git_status=$(git status --porcelain "$file_path" 2>/dev/null)
        if [[ -n "$git_status" ]]; then
            log_message "INFO" "Git status: $git_status"
            # Log first 10 lines of diff
            git diff "$file_path" 2>/dev/null | head -10 >> "$LOG_FILE"
        fi
    fi
    
    # Create audit summary
    local audit_summary=$(cat <<EOF
{
  "timestamp": "$(date +"$DATE_FORMAT")",
  "tool": "$tool_name",
  "file": "$file_path",
  "size": $file_size,
  "lines": $file_lines,
  "sensitive_issues": "$sensitive_issues",
  "quality_issues": "$quality_issues"
}
EOF
)
    
    # Write to JSON audit log
    echo "$audit_summary" >> "$AUDIT_LOG_DIR/audit_$(date +%Y%m%d).json"
    
    return 0
}

# Parse input from Claude Code
# The hook receives JSON input via stdin
if [[ -p /dev/stdin ]]; then
    # Read JSON input
    input=$(cat)
    
    # Extract relevant fields (basic parsing - could use jq if available)
    tool_name=$(echo "$input" | grep -o '"tool"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    file_path=$(echo "$input" | grep -o '"file_path"[[:space:]]*:[[:space:]]*"[^"]*"' | cut -d'"' -f4)
    
    # If we have the required information, run the audit
    if [[ -n "$tool_name" && -n "$file_path" ]]; then
        audit_file "$tool_name" "$file_path"
    else
        log_message "ERROR" "Missing required input: tool_name=$tool_name, file_path=$file_path"
    fi
else
    # Handle command-line usage for testing
    if [[ $# -ge 2 ]]; then
        audit_file "$1" "$2"
    else
        echo "Usage: $0 <tool_name> <file_path>"
        echo "Or pipe JSON input via stdin"
        exit 1
    fi
fi

exit 0