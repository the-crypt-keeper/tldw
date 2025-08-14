# Claude Code Audit System

## Overview
This audit system automatically monitors and logs all file modifications made by Claude Code, providing security scanning, code quality checks, and compliance tracking.

## Components

### 1. Global Audit Script
**Location**: `~/.claude/audit_agent.sh`

This script runs after every file modification and:
- Logs all file changes with timestamps
- Scans for sensitive data (API keys, passwords, tokens)
- Checks code quality issues (debug statements, TODOs)
- Runs project-specific linting when available
- Creates both text and JSON audit logs

### 2. Global Settings
**Location**: `~/.claude/settings.json`

Configures hooks that trigger the audit system:
- `PostToolUse` hooks run after Write/Edit operations
- `PreToolUse` hooks log the start of modifications
- Audit logs stored in `~/.claude/audit_logs/`

### 3. Project Settings
**Location**: `.claude/settings.json`

Project-specific configuration for:
- Python linting with ruff
- Test file tracking
- Pre-commit test execution
- Custom notifications for sensitive data

### 4. Audit Configuration
**Location**: `.claude/audit_config.json`

Defines audit rules and patterns:
- Security patterns (API keys, passwords, SQL injection)
- Code quality checks (debug statements, TODOs)
- File rules (size limits, forbidden files)
- Project-specific requirements

## Audit Log Locations

- **Main log**: `~/.claude/audit_logs/audit_YYYYMMDD.log`
- **JSON log**: `~/.claude/audit_logs/audit_YYYYMMDD.json`
- **Activity log**: `~/.claude/audit_logs/activity.log`
- **Test changes**: `~/.claude/audit_logs/test_changes.log`

## Features

### Security Scanning
- Detects hardcoded credentials
- Identifies potential SQL injection vulnerabilities
- Warns about dangerous functions (eval, exec)
- Blocks commits with critical security issues

### Code Quality
- Identifies debug print statements
- Finds TODO/FIXME comments
- Detects console.log statements in JavaScript
- Runs project linters automatically

### Compliance Tracking
- Maintains audit trail of all modifications
- Tracks who modified what and when
- Generates daily summary reports
- Preserves git diff information

## Testing the System

To test if the audit system is working:

```bash
# Test the audit script directly
~/.claude/audit_agent.sh "Test" "/path/to/file.py"

# Check recent audit logs
cat ~/.claude/audit_logs/audit_$(date +%Y%m%d).log

# View JSON audit entries
cat ~/.claude/audit_logs/audit_$(date +%Y%m%d).json | jq '.'
```

## Notifications

The system can send notifications for:
- Critical security issues (macOS notifications)
- Test failures (console warnings)
- Linting errors (based on configuration)

## Customization

### Adding New Security Patterns
Edit `.claude/audit_config.json` and add patterns to the `security.patterns` array.

### Modifying Code Quality Checks
Update the `codeQuality.checks` section in the audit configuration.

### Project-Specific Rules
Add project-specific rules in the `projectSpecific` section of the audit config.

## Troubleshooting

### Audit Not Running
1. Check if the script is executable: `chmod +x ~/.claude/audit_agent.sh`
2. Verify hooks in settings.json are properly configured
3. Check for syntax errors in JSON configuration files

### Missing Logs
1. Ensure audit_logs directory exists: `mkdir -p ~/.claude/audit_logs`
2. Check file permissions on the audit script
3. Review Claude Code's hook execution logs

### False Positives
1. Adjust patterns in audit_config.json
2. Use `continueOnError: true` for non-critical checks
3. Customize severity levels for different rule types

## Maintenance

### Log Rotation
Logs are automatically organized by date. To clean old logs:
```bash
find ~/.claude/audit_logs -name "*.log" -mtime +30 -delete
```

### Updating Rules
When updating audit rules:
1. Test new patterns with sample files first
2. Monitor for false positives
3. Adjust severity levels as needed

## Integration with CI/CD

The audit system can be integrated with CI/CD pipelines:
1. Export audit logs to a central location
2. Parse JSON logs for automated reporting
3. Fail builds on critical security issues
4. Generate compliance reports from audit data