# mDNS Broadcast Service Manager for AI-Tools
# Manages the mDNS broadcaster as a scheduled task (more reliable than Windows Service for Python)
# Version: 2.0

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('install', 'uninstall', 'start', 'stop', 'status', 'logs', 'help')]
    [string]$Command = 'status'
)

# Configuration
$TaskName = "AI-Tools mDNS Broadcaster"
$TaskDescription = "Broadcasts AI-Tools services (.local domains) to the local network via mDNS"
$ScriptDir = $PSScriptRoot
$ProjectRoot = Split-Path (Split-Path $ScriptDir -Parent) -Parent
$PythonScript = Join-Path $ScriptDir "mdns_broadcast.py"
$LogFile = Join-Path $ScriptDir "mdns_service.log"

# Find Python
function Get-PythonPath {
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\pythonw.exe"
    if (Test-Path $venvPython) { return $venvPython }
    
    $venvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) { return $venvPython }
    
    return "pythonw.exe"
}

$PythonPath = Get-PythonPath

function Write-Status {
    param($Message, $Level = "INFO")
    $color = switch ($Level) {
        "ERROR" { "Red" }
        "WARN" { "Yellow" }
        "SUCCESS" { "Green" }
        default { "White" }
    }
    $symbol = switch ($Level) {
        "ERROR" { "[X]" }
        "WARN" { "[!]" }
        "SUCCESS" { "[OK]" }
        default { "[i]" }
    }
    Write-Host "$symbol $Message" -ForegroundColor $color
}

function Get-TaskInfo {
    return Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
}

function Install-MDNSTask {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  AI-Tools mDNS Broadcaster Installer" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    # Check Python
    Write-Host "[CHECK] Verifying Python..." -ForegroundColor Yellow
    if (-not (Test-Path $PythonPath)) {
        Write-Status "Python not found at: $PythonPath" "ERROR"
        return $false
    }
    Write-Status "Python: $PythonPath" "SUCCESS"

    # Check script
    if (-not (Test-Path $PythonScript)) {
        Write-Status "Script not found: $PythonScript" "ERROR"
        return $false
    }
    Write-Status "Script: mdns_broadcast.py" "SUCCESS"

    # Remove existing task
    $existing = Get-TaskInfo
    if ($existing) {
        Write-Host "[STEP 1/3] Removing existing task..." -ForegroundColor Yellow
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Status "Existing task removed" "SUCCESS"
    } else {
        Write-Host "[STEP 1/3] No existing task found" -ForegroundColor Yellow
    }

    # Create task
    Write-Host "[STEP 2/3] Creating scheduled task..." -ForegroundColor Yellow
    
    $action = New-ScheduledTaskAction -Execute $PythonPath -Argument "`"$PythonScript`"" -WorkingDirectory $ScriptDir
    $trigger = New-ScheduledTaskTrigger -AtStartup
    $principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $TaskDescription | Out-Null
    Write-Status "Task created" "SUCCESS"

    # Start task
    Write-Host "[STEP 3/3] Starting task..." -ForegroundColor Yellow
    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep 2

    $task = Get-TaskInfo
    if ($task -and $task.State -eq 'Running') {
        Write-Status "Task is running" "SUCCESS"
    } else {
        Write-Status "Task created but may need manual start" "WARN"
    }

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "mDNS services broadcasting:" -ForegroundColor Cyan
    Write-Host "  http://ai-tools.local/" -ForegroundColor White
    Write-Host "  http://audio-notes.local:7860/" -ForegroundColor White
    Write-Host "  http://lobe-chat.local:3210/" -ForegroundColor White
    Write-Host "  http://ollama.local:11434/" -ForegroundColor White

    return $true
}

function Uninstall-MDNSTask {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  AI-Tools mDNS Broadcaster Uninstaller" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    $task = Get-TaskInfo
    if (-not $task) {
        Write-Status "Task is not installed" "SUCCESS"
        return $true
    }

    Write-Host "[STEP 1/2] Stopping task..." -ForegroundColor Yellow
    if ($task.State -eq 'Running') {
        Stop-ScheduledTask -TaskName $TaskName
        Start-Sleep 2
        Write-Status "Task stopped" "SUCCESS"
    } else {
        Write-Status "Task was not running" "SUCCESS"
    }

    Write-Host "[STEP 2/2] Removing task..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Status "Task removed" "SUCCESS"

    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "  Uninstallation Complete!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green

    return $true
}

function Start-MDNSTask {
    $task = Get-TaskInfo
    if (-not $task) {
        Write-Status "Task is not installed. Run: .\service-manager.ps1 install" "ERROR"
        return $false
    }
    
    if ($task.State -eq 'Running') {
        Write-Status "Task is already running" "SUCCESS"
        return $true
    }

    Start-ScheduledTask -TaskName $TaskName
    Start-Sleep 2
    
    $task = Get-TaskInfo
    if ($task.State -eq 'Running') {
        Write-Status "Task started" "SUCCESS"
    } else {
        Write-Status "Failed to start task" "ERROR"
    }
}

function Stop-MDNSTask {
    $task = Get-TaskInfo
    if (-not $task) {
        Write-Status "Task is not installed" "WARN"
        return $true
    }
    
    if ($task.State -ne 'Running') {
        Write-Status "Task is not running" "SUCCESS"
        return $true
    }

    Stop-ScheduledTask -TaskName $TaskName
    Start-Sleep 2
    Write-Status "Task stopped" "SUCCESS"
}

function Show-Status {
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host "  AI-Tools mDNS Broadcaster Status" -ForegroundColor Cyan
    Write-Host "============================================" -ForegroundColor Cyan
    Write-Host ""

    $task = Get-TaskInfo
    if (-not $task) {
        Write-Status "Task is NOT installed" "WARN"
        Write-Host ""
        Write-Host "Run: .\service-manager.ps1 install" -ForegroundColor Yellow
        return
    }

    Write-Status "Task: $TaskName" "SUCCESS"
    Write-Status "State: $($task.State)" $(if ($task.State -eq 'Running') { "SUCCESS" } else { "WARN" })
    
    $lastRun = $task.LastRunTime
    if ($lastRun -and $lastRun -ne [DateTime]::MinValue) {
        Write-Status "Last Run: $lastRun" "INFO"
    }

    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Cyan
    Write-Host "  .\service-manager.ps1 start     - Start the broadcaster"
    Write-Host "  .\service-manager.ps1 stop      - Stop the broadcaster"
    Write-Host "  .\service-manager.ps1 uninstall - Remove the task"
    Write-Host "  .\service-manager.ps1 logs      - View logs"
}

function Show-Logs {
    if (Test-Path $LogFile) {
        Get-Content $LogFile -Tail 50
    } else {
        Write-Status "No log file found" "WARN"
    }
}

function Show-Help {
    Write-Host "AI-Tools mDNS Broadcaster Service Manager" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Usage: .\service-manager.ps1 <command>" -ForegroundColor White
    Write-Host ""
    Write-Host "Commands:" -ForegroundColor Yellow
    Write-Host "  install    - Install and start the mDNS broadcaster"
    Write-Host "  uninstall  - Stop and remove the broadcaster"
    Write-Host "  start      - Start the broadcaster"
    Write-Host "  stop       - Stop the broadcaster"
    Write-Host "  status     - Show current status"
    Write-Host "  logs       - View recent logs"
    Write-Host "  help       - Show this help"
}

# Main
switch ($Command) {
    'install'   { Install-MDNSTask }
    'uninstall' { Uninstall-MDNSTask }
    'start'     { Start-MDNSTask }
    'stop'      { Stop-MDNSTask }
    'status'    { Show-Status }
    'logs'      { Show-Logs }
    'help'      { Show-Help }
    default     { Show-Status }
}
