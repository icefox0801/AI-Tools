# AI-Tools mDNS Service Manager
# Unified script for all service management operations
# Based on SystemPerformanceNotifierService service-manager.ps1
# Version: 1.0

[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateSet('install', 'uninstall', 'restart', 'start', 'stop', 'status', 'logs', 'diagnostics', 'menu', 'help')]
  [string]$Command = 'menu',

  [switch]$Force,
  [switch]$Detailed,
  [switch]$Live,
  [switch]$Quick,
  [int]$Hours = 24
)

# Global variables
$script:ServiceName = "AIToolsMDNS"
$script:ServiceDisplayName = "AI-Tools mDNS Broadcaster"
$script:ServiceDescription = "Broadcasts AI-Tools services (.local domains) to the local network via mDNS"
$script:ErrorCount = 0
$script:WarningCount = 0
$script:ScriptRoot = $PSScriptRoot
$script:ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$script:PythonScript = Join-Path $script:ScriptRoot "mdns_service.py"
$script:LogFile = Join-Path $script:ScriptRoot "mdns_service.log"

# Find Python executable
function Get-PythonPath {
  # Try venv first
  $venvPython = Join-Path $script:ProjectRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return $venvPython
  }
  
  # Try WindowsApps Python
  $windowsAppsPython = "$env:LOCALAPPDATA\Microsoft\WindowsApps\python.exe"
  if (Test-Path $windowsAppsPython) {
    return $windowsAppsPython
  }
  
  # Fall back to system Python
  return "python"
}

$script:PythonPath = Get-PythonPath

#region Utility Functions

function Test-Administrator {
  $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
  $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
  return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Request-Elevation {
  param($CommandArgs = "")

  if ($Force) {
    Write-Error "This operation requires administrator privileges. Run as administrator."
    exit 1
  }

  Write-Host "Requesting administrator privileges..." -ForegroundColor Yellow
  $arguments = "-ExecutionPolicy Bypass -File `"$PSCommandPath`" $CommandArgs -Force"
  Start-Process PowerShell -Verb RunAs -ArgumentList $arguments
  exit
}

function Write-Status {
  param($Message, $Level = "INFO")

  $color = switch ($Level) {
    "ERROR" { $script:ErrorCount++; "Red" }
    "WARN" { $script:WarningCount++; "Yellow" }
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

function Get-ServiceInfo {
  return Get-Service -Name $script:ServiceName -ErrorAction SilentlyContinue
}

function Wait-ForKeyPress {
  param($Message = "Press any key to continue...")
  Write-Host ""
  Write-Host $Message -ForegroundColor Gray
  $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

function Get-LocalIP {
  try {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 | 
      Where-Object { $_.IPAddress -match "^192\.168\." -and $_.PrefixOrigin -eq "Dhcp" } | 
      Select-Object -First 1).IPAddress
    
    if (-not $ip) {
      $ip = (Get-NetIPAddress -AddressFamily IPv4 | 
        Where-Object { $_.IPAddress -match "^192\.168\." } | 
        Select-Object -First 1).IPAddress
    }
    
    return $ip
  }
  catch {
    return "Unknown"
  }
}

#endregion

#region Service Operations

function Install-MDNSService {
  if (-not (Test-Administrator)) {
    Request-Elevation "install"
  }

  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Installer" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  try {
    # Check Python installation
    Write-Host "[CHECK] Verifying Python installation..." -ForegroundColor Yellow
    try {
      $pythonVersion = & $script:PythonPath --version 2>&1
      if ($LASTEXITCODE -eq 0) {
        Write-Status "Python: $pythonVersion" "SUCCESS"
        Write-Status "Path: $script:PythonPath" "SUCCESS"
      }
      else {
        throw "Python command failed"
      }
    }
    catch {
      Write-Status "Python not found! Install Python 3.8+ first." "ERROR"
      return $false
    }

    # Check required packages
    Write-Host "[CHECK] Verifying required packages..." -ForegroundColor Yellow
    $packages = @("zeroconf", "pywin32")
    foreach ($pkg in $packages) {
      $result = & $script:PythonPath -c "import $pkg" 2>&1
      if ($LASTEXITCODE -ne 0) {
        Write-Host "  Installing $pkg..." -ForegroundColor Yellow
        & $script:PythonPath -m pip install $pkg --quiet
      }
      Write-Status "Package: $pkg" "SUCCESS"
    }

    # Stop existing service
    Write-Host "[STEP 1/5] Stopping existing service..." -ForegroundColor Yellow
    $service = Get-ServiceInfo
    if ($service -and $service.Status -eq 'Running') {
      Stop-Service -Name $script:ServiceName -Force -ErrorAction SilentlyContinue
      Write-Status "Service stopped" "SUCCESS"
      Start-Sleep 3
    }
    else {
      Write-Status "No running service found" "SUCCESS"
    }

    # Remove existing service
    Write-Host "[STEP 2/5] Removing existing service..." -ForegroundColor Yellow
    if ($service) {
      & sc.exe delete $script:ServiceName 2>&1 | Out-Null
      Write-Status "Service removed" "SUCCESS"
      Start-Sleep 2
    }
    else {
      Write-Status "No existing service to remove" "SUCCESS"
    }

    # Verify script exists
    Write-Host "[STEP 3/5] Verifying service script..." -ForegroundColor Yellow
    if (-not (Test-Path $script:PythonScript)) {
      Write-Status "Service script not found: $script:PythonScript" "ERROR"
      return $false
    }
    Write-Status "Script found: mdns_service.py" "SUCCESS"

    # Create service using sc.exe with pythonservice.exe or nssm
    Write-Host "[STEP 4/5] Installing service..." -ForegroundColor Yellow
    
    # Use Python's pywin32 service installer
    $installResult = & $script:PythonPath $script:PythonScript install 2>&1
    if ($LASTEXITCODE -ne 0) {
      Write-Status "Service installation output: $installResult" "WARN"
    }
    
    # Configure service for auto-start
    Write-Host "[STEP 5/5] Configuring service..." -ForegroundColor Yellow
    & sc.exe config $script:ServiceName start= delayed-auto 2>&1 | Out-Null
    & sc.exe description $script:ServiceName $script:ServiceDescription 2>&1 | Out-Null
    & sc.exe failure $script:ServiceName reset= 86400 actions= restart/5000/restart/10000/restart/30000 2>&1 | Out-Null
    Write-Status "Service configured (auto-start with failure recovery)" "SUCCESS"

    # Start service
    Write-Host "[STEP 6/5] Starting service..." -ForegroundColor Yellow
    & sc.exe start $script:ServiceName 2>&1 | Out-Null
    Start-Sleep 3

    # Verify installation
    $service = Get-ServiceInfo
    if ($service -and $service.Status -eq 'Running') {
      Write-Status "Service is running" "SUCCESS"
    }
    else {
      Write-Status "Service installed but may need manual start" "WARN"
      Write-Host "  Try: net start $script:ServiceName" -ForegroundColor Yellow
    }

    # Show registered services
    $localIP = Get-LocalIP
    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host "  Installation Complete!" -ForegroundColor Green
    Write-Host "======================================================" -ForegroundColor Green
    Write-Status "Service: $script:ServiceName" "SUCCESS"
    Write-Status "Auto-start: Enabled (delayed)" "SUCCESS"
    Write-Status "Host IP: $localIP" "SUCCESS"
    Write-Host ""
    Write-Host "Registered mDNS services:" -ForegroundColor Cyan
    Write-Host "  http://ai-tools.local/" -ForegroundColor White
    Write-Host "  http://audio-notes.local:7860/" -ForegroundColor White
    Write-Host "  http://lobe-chat.local:3210/" -ForegroundColor White
    Write-Host "  http://ollama.local:11434/" -ForegroundColor White

    return $true

  }
  catch {
    Write-Status "Installation failed: $_" "ERROR"
    return $false
  }
}

function Uninstall-MDNSService {
  if (-not (Test-Administrator)) {
    Request-Elevation "uninstall"
  }

  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Uninstaller" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  try {
    $service = Get-ServiceInfo
    if (-not $service) {
      Write-Status "$script:ServiceName is not installed" "SUCCESS"
      return $true
    }

    Write-Host "[STEP 1/3] Stopping service..." -ForegroundColor Yellow
    if ($service.Status -eq 'Running') {
      Stop-Service -Name $script:ServiceName -Force

      $timeout = 0
      do {
        Start-Sleep 1
        $service = Get-ServiceInfo
        $timeout++
      } while ($service -and $service.Status -ne 'Stopped' -and $timeout -lt 10)

      if (-not $service -or $service.Status -eq 'Stopped') {
        Write-Status "Service stopped" "SUCCESS"
      }
      else {
        Write-Status "Service stop timeout (forcing removal)" "WARN"
      }
    }
    else {
      Write-Status "Service was not running" "SUCCESS"
    }

    Write-Host "[STEP 2/3] Removing service..." -ForegroundColor Yellow
    
    # Try Python uninstall first
    & $script:PythonPath $script:PythonScript remove 2>&1 | Out-Null
    
    # Also try sc.exe delete as backup
    & sc.exe delete $script:ServiceName 2>&1 | Out-Null
    
    Start-Sleep 2

    Write-Host "[STEP 3/3] Verifying removal..." -ForegroundColor Yellow
    $service = Get-ServiceInfo
    if (-not $service) {
      Write-Status "Service successfully uninstalled" "SUCCESS"
    }
    else {
      Write-Status "Service may still be visible (Windows processing removal)" "WARN"
    }

    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host "  Uninstallation Complete!" -ForegroundColor Green
    Write-Host "======================================================" -ForegroundColor Green

    return $true

  }
  catch {
    Write-Status "Uninstallation failed: $_" "ERROR"
    return $false
  }
}

function Restart-MDNSService {
  if (-not (Test-Administrator)) {
    Request-Elevation "restart"
  }

  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Restart" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  try {
    $service = Get-ServiceInfo
    if (-not $service) {
      Write-Status "$script:ServiceName is not installed!" "ERROR"
      Write-Host "Run 'service-manager.ps1 install' to install the service first." -ForegroundColor Yellow
      return $false
    }

    Write-Host "Current Status: $($service.Status)" -ForegroundColor Cyan
    Write-Host ""

    Write-Host "[STEP 1/2] Stopping service..." -ForegroundColor Yellow
    if ($service.Status -eq 'Running') {
      Stop-Service -Name $script:ServiceName -Force

      $timeout = 0
      do {
        Start-Sleep 1
        $service = Get-ServiceInfo
        $timeout++
      } while ($service -and $service.Status -ne 'Stopped' -and $timeout -lt 10)

      Write-Status "Service stopped" "SUCCESS"
    }
    else {
      Write-Status "Service was not running" "SUCCESS"
    }

    Write-Host "[STEP 2/2] Starting service..." -ForegroundColor Yellow
    Start-Service -Name $script:ServiceName

    Start-Sleep 3
    $service = Get-ServiceInfo

    if ($service -and $service.Status -eq 'Running') {
      Write-Status "Service started successfully" "SUCCESS"
    }
    else {
      Write-Status "Service restart failed - Status: $($service.Status)" "ERROR"
    }

    Write-Host ""
    Write-Host "======================================================" -ForegroundColor Green
    Write-Host "  Restart Complete!" -ForegroundColor Green
    Write-Host "======================================================" -ForegroundColor Green

    return $service -and $service.Status -eq 'Running'

  }
  catch {
    Write-Status "Restart failed: $_" "ERROR"
    return $false
  }
}

function Start-MDNSService {
  if (-not (Test-Administrator)) {
    Request-Elevation "start"
  }

  $service = Get-ServiceInfo
  if (-not $service) {
    Write-Status "$script:ServiceName is not installed!" "ERROR"
    return $false
  }

  if ($service.Status -eq 'Running') {
    Write-Status "Service is already running" "SUCCESS"
    return $true
  }

  Write-Host "Starting service..." -ForegroundColor Yellow
  Start-Service -Name $script:ServiceName
  Start-Sleep 3

  $service = Get-ServiceInfo
  if ($service.Status -eq 'Running') {
    Write-Status "Service started successfully" "SUCCESS"
    return $true
  }
  else {
    Write-Status "Failed to start service" "ERROR"
    return $false
  }
}

function Stop-MDNSService {
  if (-not (Test-Administrator)) {
    Request-Elevation "stop"
  }

  $service = Get-ServiceInfo
  if (-not $service) {
    Write-Status "$script:ServiceName is not installed!" "ERROR"
    return $false
  }

  if ($service.Status -eq 'Stopped') {
    Write-Status "Service is already stopped" "SUCCESS"
    return $true
  }

  Write-Host "Stopping service..." -ForegroundColor Yellow
  Stop-Service -Name $script:ServiceName -Force
  Start-Sleep 3

  $service = Get-ServiceInfo
  if ($service.Status -eq 'Stopped') {
    Write-Status "Service stopped successfully" "SUCCESS"
    return $true
  }
  else {
    Write-Status "Failed to stop service" "ERROR"
    return $false
  }
}

#endregion

#region Information Functions

function Show-ServiceStatus {
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Status" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  $service = Get-ServiceInfo

  if (-not $service) {
    Write-Status "Service Status: NOT INSTALLED" "ERROR"
    Write-Host ""
    Write-Host "To install the service:" -ForegroundColor Yellow
    Write-Host "  .\service-manager.ps1 install" -ForegroundColor White
    return $false
  }

  # Service status
  $statusColor = switch ($service.Status) {
    'Running' { 'Green' }
    'Stopped' { 'Red' }
    default { 'Yellow' }
  }
  Write-Host "[OK] Service Status: $($service.Status.ToString().ToUpper())" -ForegroundColor $statusColor

  # Get service configuration
  try {
    $config = & sc.exe qc $script:ServiceName 2>&1 | Out-String
    $startType = ($config | Select-String "START_TYPE").ToString().Split(":")[1].Trim()
    Write-Host "[OK] Start Type: $startType" -ForegroundColor Green
  }
  catch {
    Write-Host "[!] Unable to get service configuration" -ForegroundColor Yellow
  }

  # Network info
  $localIP = Get-LocalIP
  Write-Host "[OK] Host IP: $localIP" -ForegroundColor Green

  # mDNS Services
  Write-Host ""
  Write-Host "Registered mDNS Services:" -ForegroundColor Cyan
  Write-Host "========================" -ForegroundColor Cyan
  Write-Host "  ai-tools.local:80" -ForegroundColor White
  Write-Host "  audio-notes.local:7860" -ForegroundColor White
  Write-Host "  lobe-chat.local:3210" -ForegroundColor White
  Write-Host "  ollama.local:11434" -ForegroundColor White

  # Show log file info
  if (Test-Path $script:LogFile) {
    $logInfo = Get-Item $script:LogFile
    Write-Host ""
    Write-Host "[OK] Log File: $($logInfo.Name) ($([math]::Round($logInfo.Length / 1KB, 1)) KB)" -ForegroundColor Green
  }

  return $true
}

function Show-ServiceLogs {
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Logs" -ForegroundColor Cyan
  if ($Live) {
    Write-Host "  Live Monitoring Mode" -ForegroundColor Yellow
  }
  else {
    Write-Host "  Last $Hours hours" -ForegroundColor Yellow
  }
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  # Check if log file exists
  if (-not (Test-Path $script:LogFile)) {
    Write-Host "[!] Log file not found: $script:LogFile" -ForegroundColor Yellow
    Write-Host "The service may not have started yet." -ForegroundColor Gray
    return
  }

  if ($Live) {
    Write-Host "Starting live monitoring (Press Ctrl+C to stop)..." -ForegroundColor Yellow
    Write-Host ""
    Get-Content $script:LogFile -Wait -Tail 20
    return
  }

  # Show recent log entries
  $logs = Get-Content $script:LogFile -Tail 50
  foreach ($line in $logs) {
    $color = "White"
    if ($line -match "\[ERROR\]") { $color = "Red" }
    elseif ($line -match "\[WARNING\]") { $color = "Yellow" }
    elseif ($line -match "\[INFO\]") { $color = "Green" }
    Write-Host $line -ForegroundColor $color
  }
}

function Start-ServiceDiagnostics {
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Diagnostics" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  $script:ErrorCount = 0
  $script:WarningCount = 0

  # Python Check
  Write-Host "Python Environment:" -ForegroundColor Cyan
  Write-Host "==================" -ForegroundColor Cyan

  try {
    $pythonVersion = & $script:PythonPath --version 2>&1
    Write-Status "Python: $pythonVersion" "SUCCESS"
    Write-Status "Path: $script:PythonPath" "SUCCESS"
  }
  catch {
    Write-Status "Python: Not Found" "ERROR"
  }

  # Package Check
  Write-Host ""
  Write-Host "Required Packages:" -ForegroundColor Cyan
  Write-Host "=================" -ForegroundColor Cyan

  $packages = @("zeroconf", "win32serviceutil")
  foreach ($pkg in $packages) {
    $result = & $script:PythonPath -c "import $pkg" 2>&1
    if ($LASTEXITCODE -eq 0) {
      Write-Status "$pkg" "SUCCESS"
    }
    else {
      Write-Status "$pkg - Not Installed" "ERROR"
    }
  }

  # Service Check
  Write-Host ""
  Write-Host "Service Status:" -ForegroundColor Cyan
  Write-Host "==============" -ForegroundColor Cyan

  $service = Get-ServiceInfo
  if (-not $service) {
    Write-Status "Installation: Not Installed" "ERROR"
  }
  else {
    Write-Status "Installation: Installed" "SUCCESS"
    
    switch ($service.Status) {
      'Running' { Write-Status "Status: Running" "SUCCESS" }
      'Stopped' { Write-Status "Status: Stopped" "WARN" }
      default { Write-Status "Status: $($service.Status)" "WARN" }
    }
  }

  # Network Check
  Write-Host ""
  Write-Host "Network Status:" -ForegroundColor Cyan
  Write-Host "==============" -ForegroundColor Cyan

  $localIP = Get-LocalIP
  if ($localIP -and $localIP -ne "Unknown") {
    Write-Status "Local IP: $localIP" "SUCCESS"
  }
  else {
    Write-Status "Local IP: Could not detect" "WARN"
  }

  # mDNS Port Check
  try {
    $udp5353 = Get-NetUDPEndpoint -LocalPort 5353 -ErrorAction SilentlyContinue
    if ($udp5353) {
      Write-Status "mDNS Port 5353: In Use" "SUCCESS"
    }
    else {
      Write-Status "mDNS Port 5353: Available" "SUCCESS"
    }
  }
  catch {
    Write-Status "mDNS Port: Unable to check" "WARN"
  }

  # Summary
  Write-Host ""
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  Diagnostic Summary" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan

  if ($script:ErrorCount -eq 0 -and $script:WarningCount -eq 0) {
    Write-Host "[SUCCESS] ALL CHECKS PASSED - System is healthy!" -ForegroundColor Green
  }
  else {
    if ($script:ErrorCount -gt 0) {
      Write-Host "[X] ERRORS FOUND: $script:ErrorCount" -ForegroundColor Red
    }
    if ($script:WarningCount -gt 0) {
      Write-Host "[!] WARNINGS: $script:WarningCount" -ForegroundColor Yellow
    }
  }
}

#endregion

#region Menu System

function Show-MainMenu {
  Clear-Host
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Manager v1.0" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  # Show current status
  $service = Get-ServiceInfo
  if ($service) {
    $statusColor = switch ($service.Status) {
      'Running' { 'Green' }
      'Stopped' { 'Red' }
      default { 'Yellow' }
    }
    Write-Host "Current Status: $($service.Status.ToString().ToUpper())" -ForegroundColor $statusColor
    
    $localIP = Get-LocalIP
    Write-Host "Host IP: $localIP" -ForegroundColor Cyan
  }
  else {
    Write-Host "Current Status: NOT INSTALLED" -ForegroundColor Red
  }

  Write-Host ""
  Write-Host "Service Operations:" -ForegroundColor Yellow
  Write-Host ""

  if (-not $service) {
    Write-Host "  [1] Install Service" -ForegroundColor White
  }
  else {
    Write-Host "  [1] Restart Service" -ForegroundColor White
    Write-Host "  [2] Stop Service" -ForegroundColor White
    Write-Host "  [3] Start Service" -ForegroundColor White
    Write-Host "  [4] Reinstall Service" -ForegroundColor White
    Write-Host "  [5] Uninstall Service" -ForegroundColor White
  }

  Write-Host ""
  Write-Host "Information & Diagnostics:" -ForegroundColor Yellow
  Write-Host "  [S] Service Status" -ForegroundColor White
  Write-Host "  [L] View Logs" -ForegroundColor White
  Write-Host "  [D] Run Diagnostics" -ForegroundColor White
  Write-Host "  [M] Live Log Monitoring" -ForegroundColor White

  Write-Host ""
  Write-Host "Other Options:" -ForegroundColor Yellow
  Write-Host "  [H] Help" -ForegroundColor White
  Write-Host "  [Q] Quit" -ForegroundColor White

  Write-Host ""
}

function Show-HelpInfo {
  Clear-Host
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host "  AI-Tools mDNS Service Manager Help" -ForegroundColor Cyan
  Write-Host "======================================================" -ForegroundColor Cyan
  Write-Host ""

  Write-Host "Command Line Usage:" -ForegroundColor Yellow
  Write-Host "  .\service-manager.ps1 <command> [options]" -ForegroundColor White
  Write-Host ""

  Write-Host "Commands:" -ForegroundColor Yellow
  Write-Host "  install       Install service with auto-start" -ForegroundColor White
  Write-Host "  uninstall     Remove service completely" -ForegroundColor White
  Write-Host "  start         Start the service" -ForegroundColor White
  Write-Host "  stop          Stop the service" -ForegroundColor White
  Write-Host "  restart       Restart the service" -ForegroundColor White
  Write-Host "  status        Show service status" -ForegroundColor White
  Write-Host "  logs          View service logs" -ForegroundColor White
  Write-Host "  diagnostics   Run system diagnostics" -ForegroundColor White
  Write-Host "  menu          Show interactive menu (default)" -ForegroundColor White
  Write-Host "  help          Show this help" -ForegroundColor White
  Write-Host ""

  Write-Host "Options:" -ForegroundColor Yellow
  Write-Host "  -Detailed     Show detailed information" -ForegroundColor White
  Write-Host "  -Live         Live log monitoring mode" -ForegroundColor White
  Write-Host "  -Quick        Quick diagnostics mode" -ForegroundColor White
  Write-Host "  -Hours <n>    Show logs for last n hours (default: 24)" -ForegroundColor White
  Write-Host "  -Force        Force operation without prompts" -ForegroundColor White
  Write-Host ""

  Write-Host "Examples:" -ForegroundColor Yellow
  Write-Host "  .\service-manager.ps1 install" -ForegroundColor Green
  Write-Host "  .\service-manager.ps1 status" -ForegroundColor Green
  Write-Host "  .\service-manager.ps1 logs -Live" -ForegroundColor Green
  Write-Host ""

  Write-Host "mDNS Services Broadcast:" -ForegroundColor Yellow
  Write-Host "  http://ai-tools.local/" -ForegroundColor White
  Write-Host "  http://audio-notes.local:7860/" -ForegroundColor White
  Write-Host "  http://lobe-chat.local:3210/" -ForegroundColor White
  Write-Host "  http://ollama.local:11434/" -ForegroundColor White

  if ($Command -eq 'menu') {
    Write-Host ""
    Write-Host "Press any key to return to menu..." -ForegroundColor Gray
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
  }
}

function Start-InteractiveMenu {
  while ($true) {
    Show-MainMenu
    Write-Host "Select option: " -ForegroundColor Cyan -NoNewline
    $choice = Read-Host
    $choice = $choice.ToUpper()

    $service = Get-ServiceInfo

    switch ($choice) {
      "1" {
        if (-not $service) {
          Install-MDNSService
        }
        else {
          Restart-MDNSService
        }
        Wait-ForKeyPress
      }

      "2" {
        if ($service) {
          Stop-MDNSService
          Wait-ForKeyPress
        }
      }

      "3" {
        if ($service) {
          Start-MDNSService
          Wait-ForKeyPress
        }
      }

      "4" {
        if ($service) {
          Install-MDNSService
          Wait-ForKeyPress
        }
      }

      "5" {
        if ($service) {
          Uninstall-MDNSService
          Wait-ForKeyPress
        }
      }

      "S" {
        Show-ServiceStatus
        Wait-ForKeyPress
      }

      "L" {
        Show-ServiceLogs
        Wait-ForKeyPress
      }

      "D" {
        Start-ServiceDiagnostics
        Wait-ForKeyPress
      }

      "M" {
        Write-Host ""
        Write-Host "Starting live monitoring (Press Ctrl+C to stop)..." -ForegroundColor Yellow
        $script:Live = $true
        Show-ServiceLogs
      }

      "H" { Show-HelpInfo }

      "Q" {
        Write-Host ""
        Write-Host "Goodbye!" -ForegroundColor Green
        exit 0
      }

      default {
        Write-Host ""
        Write-Host "Invalid option. Please try again." -ForegroundColor Red
        Start-Sleep 2
      }
    }
  }
}

#endregion

#region Main Execution

switch ($Command.ToLower()) {
  'install' {
    Install-MDNSService
    if ($Force) { Wait-ForKeyPress }
  }

  'uninstall' {
    Uninstall-MDNSService
    if ($Force) { Wait-ForKeyPress }
  }

  'start' {
    Start-MDNSService
    if ($Force) { Wait-ForKeyPress }
  }

  'stop' {
    Stop-MDNSService
    if ($Force) { Wait-ForKeyPress }
  }

  'restart' {
    Restart-MDNSService
    if ($Force) { Wait-ForKeyPress }
  }

  'status' {
    Show-ServiceStatus
    if ($Force) { Wait-ForKeyPress }
  }

  'logs' {
    Show-ServiceLogs
    if ($Force -and -not $Live) { Wait-ForKeyPress }
  }

  'diagnostics' {
    Start-ServiceDiagnostics
    if ($Force) { Wait-ForKeyPress }
  }

  'help' {
    Show-HelpInfo
  }

  'menu' {
    Start-InteractiveMenu
  }

  default {
    Start-InteractiveMenu
  }
}

#endregion
