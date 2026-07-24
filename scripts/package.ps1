param([switch]$SkipVerify)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Push-Location $RepositoryRoot
try {
    if (-not $SkipVerify) {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\verify.ps1
        if ($LASTEXITCODE -ne 0) {
            throw "Verification failed."
        }
    }

    $VersionLine = Select-String `
        -Path "services\api\local_lm\__init__.py" `
        -Pattern '^__version__ = "([^"]+)"$'
    $Version = $VersionLine.Matches[0].Groups[1].Value
    if (-not $Version) {
        throw "Could not determine the LM Atelier version."
    }

    & npm.cmd run build
    if ($LASTEXITCODE -ne 0) {
        throw "The web build failed."
    }

    $ReleaseRoot = New-Item -ItemType Directory -Force -Path "release"
    $StagingRoot = Join-Path ([IO.Path]::GetTempPath()) "lm-atelier-package-$([guid]::NewGuid())"
    $BundleName = "lm-atelier-$Version"
    $BundleRoot = Join-Path $StagingRoot $BundleName
    $SourceArchive = Join-Path $StagingRoot "source.zip"
    $TarPath = Join-Path $ReleaseRoot.FullName "$BundleName.tar.gz"
    $ZipPath = Join-Path $ReleaseRoot.FullName "$BundleName.zip"

    try {
        New-Item -ItemType Directory -Force -Path $StagingRoot | Out-Null
        & git.exe archive --format=zip "--output=$SourceArchive" HEAD
        if ($LASTEXITCODE -ne 0) {
            throw "Could not export the tracked source tree."
        }
        Expand-Archive -LiteralPath $SourceArchive -DestinationPath $BundleRoot
        New-Item -ItemType Directory -Force -Path (Join-Path $BundleRoot "apps\web") | Out-Null
        Copy-Item -Recurse -Force "apps\web\dist" (Join-Path $BundleRoot "apps\web\dist")

        Remove-Item -Force -ErrorAction SilentlyContinue $TarPath, $ZipPath
        & tar.exe -C $StagingRoot -czf $TarPath $BundleName
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the Linux release archive."
        }

        & powershell.exe `
            -NoProfile `
            -ExecutionPolicy Bypass `
            -File .\packaging\windows\build-launchers.ps1 `
            -OutputDirectory $BundleRoot
        if ($LASTEXITCODE -ne 0) {
            throw "Could not build the Windows launchers."
        }
        & tar.exe -C $StagingRoot -a -cf $ZipPath $BundleName
        if ($LASTEXITCODE -ne 0) {
            throw "Could not create the Windows release archive."
        }

        $ChecksumLines = @($TarPath, $ZipPath) | ForEach-Object {
            $Hash = (Get-FileHash -Algorithm SHA256 -LiteralPath $_).Hash.ToLowerInvariant()
            "$Hash  $([IO.Path]::GetFileName($_))"
        }
        [IO.File]::WriteAllLines(
            (Join-Path $ReleaseRoot.FullName "SHA256SUMS"),
            $ChecksumLines,
            [Text.UTF8Encoding]::new($false)
        )
    }
    finally {
        if (Test-Path -LiteralPath $StagingRoot) {
            Remove-Item -Recurse -Force -LiteralPath $StagingRoot
        }
    }

    Write-Host "Created $TarPath and $ZipPath"
}
finally {
    Pop-Location
}
