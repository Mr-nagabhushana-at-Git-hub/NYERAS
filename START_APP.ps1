param(
  [string]$RepoRoot = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$AppPath = Join-Path $RepoRoot 'retail_ui_server.py'
$RequirementsPath = Join-Path $RepoRoot 'requirements.txt'
$BrowserPort = 7860
$MinicondaRoot = Join-Path $env:LOCALAPPDATA 'miniconda3'
$BootstrapRoot = Join-Path $env:LOCALAPPDATA 'NYERAS'
$EnvPath = Join-Path $BootstrapRoot 'retail-ui'
$MarkerPath = Join-Path $EnvPath '.requirements.sha256'
$InstallerUrl = 'https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe'
$InstallerPath = Join-Path $env:TEMP 'Miniconda3-latest-Windows-x86_64.exe'

function Write-Step {
  param([Parameter(Mandatory = $true)][string]$Message)
  Write-Host "[start] $Message" -ForegroundColor Cyan
}

function Get-CondaExe {
  $candidates = @(
    $env:CONDA_EXE,
    (Join-Path $MinicondaRoot 'Scripts\conda.exe'),
    'C:\Anaconda3NEW2025\anaconda3\Scripts\conda.exe',
    (Join-Path $env:USERPROFILE 'miniconda3\Scripts\conda.exe'),
    'C:\Miniconda3\Scripts\conda.exe'
  ) | Where-Object { $_ }

  foreach ($candidate in $candidates) {
    if (Test-Path $candidate) {
      return (Resolve-Path $candidate).Path
    }
  }

  $command = Get-Command conda -ErrorAction SilentlyContinue
  if ($command -and $command.Source) {
    return $command.Source
  }

  return $null
}

function Install-Miniconda {
  if (Test-Path (Join-Path $MinicondaRoot 'Scripts\conda.exe')) {
    return
  }

  Write-Step 'Conda was not found. Downloading Miniconda...'
  [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
  if (Test-Path $InstallerPath) {
    Remove-Item $InstallerPath -Force
  }
  Invoke-WebRequest -Uri $InstallerUrl -OutFile $InstallerPath

  Write-Step "Installing Miniconda to $MinicondaRoot"
  New-Item -ItemType Directory -Force -Path (Split-Path $MinicondaRoot -Parent) | Out-Null
  $arguments = @(
    '/S',
    '/InstallationType=JustMe',
    '/AddToPath=0',
    '/RegisterPython=0',
    "/D=$MinicondaRoot"
  )
  Start-Process -FilePath $InstallerPath -ArgumentList $arguments -Wait -NoNewWindow
}

function Ensure-ProjectEnvironment {
  param(
    [Parameter(Mandatory = $true)][string]$CondaExe
  )

  if (-not (Test-Path $AppPath)) {
    throw "Could not find retail_ui_server.py at $AppPath"
  }
  if (-not (Test-Path $RequirementsPath)) {
    throw "Could not find requirements.txt at $RequirementsPath"
  }

  $requirementsHash = (Get-FileHash -Path $RequirementsPath -Algorithm SHA256).Hash
  $pythonExe = Join-Path $EnvPath 'python.exe'
  $needsCreate = -not (Test-Path $pythonExe)
  $storedHash = $null
  if (Test-Path $MarkerPath) {
    $storedHash = (Get-Content -Path $MarkerPath -ErrorAction SilentlyContinue | Select-Object -First 1)
  }
  $needsPackages = $needsCreate -or ($storedHash -ne $requirementsHash)

  if ($needsCreate) {
    Write-Step "Creating project environment at $EnvPath"
    New-Item -ItemType Directory -Force -Path $BootstrapRoot | Out-Null
    & $CondaExe create -y -p $EnvPath python pip
    if ($LASTEXITCODE -ne 0) {
      throw 'Failed to create the conda environment.'
    }
  }

  if ($needsPackages) {
    Write-Step 'Installing Python dependencies from requirements.txt'
    & $CondaExe run -p $EnvPath --no-capture-output python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) {
      throw 'Failed to upgrade pip in the project environment.'
    }
    & $CondaExe run -p $EnvPath --no-capture-output python -m pip install -r $RequirementsPath
    if ($LASTEXITCODE -ne 0) {
      throw 'Failed to install the project requirements.'
    }
    Set-Content -Path $MarkerPath -Value $requirementsHash -Encoding ASCII
  } else {
    Write-Step 'Project environment is already up to date.'
  }
}

function Start-BrowserWatcher {
  param([Parameter(Mandatory = $true)][int]$Port)

  Start-Job -ScriptBlock {
    param($JobPort)
    while (-not (Test-NetConnection -ComputerName 127.0.0.1 -Port $JobPort -InformationLevel Quiet)) {
      Start-Sleep -Milliseconds 400
    }
    Start-Process "http://127.0.0.1:$JobPort/"
  } -ArgumentList $Port
}

Write-Step "Project root: $RepoRoot"
$CondaExe = Get-CondaExe
if (-not $CondaExe) {
  Install-Miniconda
  $CondaExe = Get-CondaExe
}
if (-not $CondaExe) {
  throw 'Conda could not be located after bootstrap.'
}

Write-Step "Using conda executable: $CondaExe"
Ensure-ProjectEnvironment -CondaExe $CondaExe

$browserJob = Start-BrowserWatcher -Port $BrowserPort
Write-Step "Browser will open automatically at http://127.0.0.1:$BrowserPort/"
Write-Step 'Starting the dashboard in this console so the logs stay visible.'

$exitCode = 0
try {
  & $CondaExe run -p $EnvPath --no-capture-output python $AppPath
  $exitCode = $LASTEXITCODE
} finally {
  if ($browserJob) {
    Stop-Job $browserJob -ErrorAction SilentlyContinue | Out-Null
    Remove-Job $browserJob -Force -ErrorAction SilentlyContinue | Out-Null
  }
}

if ($exitCode -ne 0) {
  throw "Dashboard process exited with code $exitCode."
}
