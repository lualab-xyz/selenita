# Selenita alpha installer (Windows PowerShell 5.1+ / pwsh).
#
# Usage:
#   irm https://raw.githubusercontent.com/lualab-xyz/selenita/alpha-1/install.ps1 | iex
#   pwsh ./install.ps1 [version]   # e.g. 0.1.0a1 — default: newest release
#
# Behavior: resolves the release tag, finds a local Python >= 3.11, installs
# the release wheel into a dedicated venv (%USERPROFILE%\.selenita\venv),
# creates a `selenita.cmd` shim in %USERPROFILE%\.selenita\bin and adds that
# directory to the user PATH.
param([string]$Version)

$ErrorActionPreference = "Stop"
$Repo = "lualab-xyz/selenita"
$Prefix = if ($env:SELENITA_HOME) { $env:SELENITA_HOME } else { Join-Path $env:USERPROFILE ".selenita" }

# --- 1. Resolve the release tag (prereleases included via the release list) ---
if ($Version) {
    $Tag = "v" + $Version.TrimStart("v")
}
else {
    $Releases = Invoke-RestMethod "https://api.github.com/repos/$Repo/releases?per_page=1"
    if (-not $Releases) { throw "No releases found in https://github.com/$Repo" }
    $Tag = $Releases[0].tag_name
}
$Wheel = "selenita-$($Tag.TrimStart('v'))-py3-none-any.whl"
Write-Host "==> Installing selenita $($Tag.TrimStart('v'))"

# --- 2. Locate a Python >= 3.11 ------------------------------------------------
$Py = $null
foreach ($Candidate in @("py", "python3", "python")) {
    if (Get-Command $Candidate -ErrorAction SilentlyContinue) {
        $Ver = (& $Candidate -c "import sys; print('%d.%d' % sys.version_info[:2])") 2>$null
        if ($Ver) {
            $Parts = $Ver.Split(".")
            if ([int]$Parts[0] -ge 3 -and [int]$Parts[1] -ge 11) { $Py = $Candidate; break }
        }
    }
}
if (-not $Py) { throw "Python >= 3.11 is required (install from https://www.python.org/downloads/)" }
$PyVersion = (& $Py -c "import sys; print(sys.version.split()[0])")
Write-Host "==> Using Python: $PyVersion"

# --- 3. Install into a dedicated venv ------------------------------------------
New-Item -ItemType Directory -Force -Path $Prefix | Out-Null
$WheelUri = "https://github.com/$Repo/releases/download/$Tag/$Wheel"
& $Py -m venv (Join-Path $Prefix "venv")
& (Join-Path $Prefix "venv\Scripts\python.exe") -m pip install --quiet --upgrade $WheelUri
if ($LASTEXITCODE -ne 0) { throw "pip install failed — does the release exist? (https://github.com/$Repo/releases)" }

# --- 4. Create the selenita command shim and expose it on PATH ------------------
$BinDir = Join-Path $Prefix "bin"
$ShimPath = Join-Path $BinDir "selenita.cmd"
New-Item -ItemType Directory -Force -Path $BinDir | Out-Null
Set-Content -Path $ShimPath -Value "@`"$Prefix\venv\Scripts\selenita.exe`" %*" -Encoding ASCII

$UserPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($UserPath -notlike "*$BinDir*") {
    [Environment]::SetEnvironmentVariable("Path", "$UserPath;$BinDir", "User")
    Write-Host "==> Added $BinDir to your user PATH (open a new terminal to use it)"
}

Write-Host "==> Try: selenita --help   (configure the LLM endpoint via SELENITA_* env vars)"