# Comprehensive Test script for TLDW installer and runner
# Save this as test_tldw_install.ps1

$ErrorActionPreference = "Stop"

function Test-Command {
    param (
        [string]$Name,
        [scriptblock]$Command
    )
    Write-Host "Running test: $Name"
    try {
        & $Command
        Write-Host "Test passed: $Name" -ForegroundColor Green
    }
    catch {
        Write-Host "Test failed: $Name" -ForegroundColor Red
        Write-Host "Error: $_"
        exit 1
    }
}

function Test-Dependency {
    param (
        [string]$Name,
        [string]$Command
    )
    Write-Host "Checking dependency: $Name"
    if (Get-Command $Command -ErrorAction SilentlyContinue) {
        Write-Host "$Name is installed" -ForegroundColor Green
    } else {
        Write-Host "$Name is not installed" -ForegroundColor Red
        exit 1
    }
}

# Check dependencies
Test-Dependency "Git" "git"
Test-Dependency "Python" "python"

# Set up test environment
$testDir = ".\tldw_test"
$tldwDir = Join-Path $testDir "tldw"

# Cleanup function
function Cleanup-TestEnvironment {
    if (Test-Path $testDir) {
        Remove-Item -Recurse -Force $testDir
        if (Test-Path $testDir) {
            throw "Failed to clean up test environment"
        }
    }
}

# Ensure idempotency by cleaning up before and after tests
Cleanup-TestEnvironment

New-Item -ItemType Directory -Force -Path $testDir | Out-Null
Set-Location $testDir

# Copy installer and runner scripts
Copy-Item ..\Windows_Install_Update.bat .\
Copy-Item ..\Windows_Run_tldw.bat .\

# Test installation
Test-Command "Installation Script Execution" {
    $process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "echo n | Windows_Install_Update.bat" -NoNewWindow -PassThru -Wait -RedirectStandardOutput "install_output.txt" -RedirectStandardError "install_error.txt"
    if ($process.ExitCode -ne 0) {
        $errorContent = Get-Content "install_error.txt"
        throw "Installation script failed with exit code $($process.ExitCode). Error: $errorContent"
    }
}

Test-Command "TLDW Directory Creation" {
    if (-not (Test-Path $tldwDir)) {
        throw "TLDW directory was not created"
    }
}

Test-Command "Virtual Environment Creation" {
    if (-not (Test-Path (Join-Path $tldwDir "venv"))) {
        throw "Virtual environment was not created"
    }
}

Test-Command "Required Files Presence" {
    $requiredFiles = @("summarize.py", "config.txt", "ffmpeg.exe")
    foreach ($file in $requiredFiles) {
        if (-not (Test-Path (Join-Path $tldwDir $file))) {
            throw "Required file $file is missing"
        }
    }
}

# Configuration file validation
Test-Command "Configuration File Validation" {
    $configPath = Join-Path $tldwDir "config.txt"
    $configContent = Get-Content $configPath -Raw
    if (-not ($configContent -match "device\s*=\s*(cuda|cpu)")) {
        throw "Config file does not contain valid device setting"
    }
    # Add more config checks as needed
}

# Test running script with timeout
Test-Command "Run Script Execution" {
    $runProcess = Start-Process -FilePath "cmd.exe" -ArgumentList "/c", "Windows_Run_tldw.bat" -NoNewWindow -PassThru

    # Wait for the process to start
    Start-Sleep -Seconds 5

    # Check if the process is running
    $tldwProcess = Get-Process | Where-Object { $_.Name -eq "python" -and $_.CommandLine -like "*summarize.py*" }

    if (-not $tldwProcess) {
        throw "TLDW process did not start"
    }

    # Wait for 30 seconds or until the process exits
    $exited = $runProcess.WaitForExit(30000)

    if (-not $exited) {
        # If the process didn't exit, kill it
        $runProcess.Kill()
        $tldwProcess.Kill()
    }

    if ($runProcess.ExitCode -ne 0) {
        throw "Run script failed with exit code $($runProcess.ExitCode)"
    }
}

# Return to original directory
Set-Location ..

# Cleanup and verify
Test-Command "Cleanup and Verification" {
    Cleanup-TestEnvironment
}

Write-Host "All tests passed successfully!" -ForegroundColor Green