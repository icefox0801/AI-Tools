#!/usr/bin/env pwsh
# Force kill and restart Docker Desktop

Write-Host 'Stopping all Docker processes...' -ForegroundColor Yellow

# Kill all Docker-related processes
$dockerProcesses = @(
    "Docker Desktop",
    "com.docker.backend",
    "com.docker.service",
    "com.docker.proxy",
    "com.docker.wsl-distro-proxy",
    "com.docker.cli",
    "com.docker.dev-envs",
    "com.docker.extension-manager",
    "com.docker.extensions-desktop-backend",
    "docker-index",
    "vpnkit",
    "dockerd"
)

foreach ($process in $dockerProcesses) {
    $running = Get-Process -Name $process -ErrorAction SilentlyContinue
    if ($running) {
        Write-Host "  Stopping $process..." -ForegroundColor Gray
        Stop-Process -Name $process -Force -ErrorAction SilentlyContinue
    }
}

Write-Host 'Waiting for complete cleanup...' -ForegroundColor Gray
Start-Sleep -Seconds 3

Write-Host 'Starting Docker Desktop...' -ForegroundColor Green
Start-Process -FilePath "C:\Program Files\Docker\Docker\Docker Desktop.exe"

Write-Host 'Docker Desktop is starting... Please wait for it to be ready.' -ForegroundColor Cyan
Write-Host 'You may need to wait 30-60 seconds for all services to initialize.' -ForegroundColor Gray
