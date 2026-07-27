[CmdletBinding()]
param(
    [switch]$Apply,
    [string]$Repository = "ajccarlson/lm-atelier"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ApiVersion = "2026-03-10"
$ExpectedOwnerId = 32660587
$AllowedActionPatterns = @(
    "actions/attest@f7c74d28b9d84cb8768d0b8ca14a4bac6ef463e6",
    "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1",
    "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c",
    "actions/setup-node@820762786026740c76f36085b0efc47a31fe5020",
    "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97",
    "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a",
    "astral-sh/setup-uv@c771a70e6277c0a99b617c7a806ffedaca235ff9"
)

function Invoke-GitHubApi {
    param(
        [Parameter(Mandatory)]
        [ValidateSet("GET", "PATCH", "POST", "PUT")]
        [string]$Method,
        [Parameter(Mandatory)]
        [string]$Endpoint,
        [object]$Body
    )

    $Arguments = @(
        "api",
        "--method", $Method,
        "-H", "Accept: application/vnd.github+json",
        "-H", "X-GitHub-Api-Version: $ApiVersion",
        $Endpoint
    )
    if ($null -eq $Body) {
        $Output = & gh @Arguments
    } else {
        $Json = $Body | ConvertTo-Json -Depth 20 -Compress
        $BodyFile = [System.IO.Path]::GetTempFileName()
        try {
            [System.IO.File]::WriteAllText($BodyFile, $Json)
            $Output = & gh @Arguments --input $BodyFile
        } finally {
            Remove-Item -LiteralPath $BodyFile -Force
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "GitHub API request failed: $Method $Endpoint"
    }
    return ($Output -join [Environment]::NewLine)
}

function Read-JsonFile {
    param([Parameter(Mandatory)][string]$Path)

    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
}

function Test-JsonSubset {
    param(
        [AllowNull()]
        [object]$Actual,
        [AllowNull()]
        [object]$Expected
    )

    if ($null -eq $Expected) {
        return $null -eq $Actual
    }
    if ($null -eq $Actual) {
        return $false
    }

    if ($Expected -is [System.Collections.IDictionary]) {
        foreach ($Key in $Expected.Keys) {
            $ActualValue = if ($Actual -is [System.Collections.IDictionary]) {
                if (-not $Actual.Contains($Key)) {
                    return $false
                }
                $Actual[$Key]
            } else {
                $Property = $Actual.PSObject.Properties[[string]$Key]
                if (-not $Property) {
                    return $false
                }
                $Property.Value
            }
            if (-not (Test-JsonSubset -Actual $ActualValue -Expected $Expected[$Key])) {
                return $false
            }
        }
        return $true
    }

    if ($Expected -is [PSCustomObject]) {
        foreach ($Property in $Expected.PSObject.Properties) {
            $ActualProperty = $Actual.PSObject.Properties[$Property.Name]
            if (
                -not $ActualProperty -or
                -not (
                    Test-JsonSubset `
                        -Actual $ActualProperty.Value `
                        -Expected $Property.Value
                )
            ) {
                return $false
            }
        }
        return $true
    }

    $ExpectedIsSequence =
        $Expected -is [System.Collections.IEnumerable] -and
        $Expected -isnot [string]
    if ($ExpectedIsSequence) {
        if (
            $Actual -isnot [System.Collections.IEnumerable] -or
            $Actual -is [string]
        ) {
            return $false
        }
        [object[]]$ExpectedItems = @($Expected)
        [object[]]$ActualItems = @($Actual)
        if ($ExpectedItems.Count -ne $ActualItems.Count) {
            return $false
        }
        [bool[]]$Matched = [bool[]]::new($ActualItems.Count)
        foreach ($ExpectedItem in $ExpectedItems) {
            $Found = $false
            for ($Index = 0; $Index -lt $ActualItems.Count; $Index++) {
                if (
                    -not $Matched[$Index] -and
                    (
                        Test-JsonSubset `
                            -Actual $ActualItems[$Index] `
                            -Expected $ExpectedItem
                    )
                ) {
                    $Matched[$Index] = $true
                    $Found = $true
                    break
                }
            }
            if (-not $Found) {
                return $false
            }
        }
        return $true
    }

    return $Actual -eq $Expected
}

function Assert-JsonSubset {
    param(
        [Parameter(Mandatory)]
        [string]$Label,
        [AllowNull()]
        [object]$Actual,
        [AllowNull()]
        [object]$Expected
    )

    if (-not (Test-JsonSubset -Actual $Actual -Expected $Expected)) {
        throw "$Label does not match the reviewed public configuration."
    }
}

function Set-Ruleset {
    param([Parameter(Mandatory)][object]$Definition)

    $Rulesets = Invoke-GitHubApi -Method GET -Endpoint "repos/$Repository/rulesets" |
        ConvertFrom-Json
    $Existing = $Rulesets | Where-Object { $_.name -eq $Definition.name } |
        Select-Object -First 1
    if ($Existing) {
        Invoke-GitHubApi `
            -Method PUT `
            -Endpoint "repos/$Repository/rulesets/$($Existing.id)" `
            -Body $Definition | Out-Null
        Write-Host "Updated ruleset: $($Definition.name)"
    } else {
        Invoke-GitHubApi `
            -Method POST `
            -Endpoint "repos/$Repository/rulesets" `
            -Body $Definition | Out-Null
        Write-Host "Created ruleset: $($Definition.name)"
    }
}

if ($Repository -ne "ajccarlson/lm-atelier") {
    throw "Refusing to configure an unexpected repository: $Repository"
}

$BranchRules = Read-JsonFile (
    Join-Path $RepositoryRoot ".github\rulesets\public-branches.json"
)
$TagRules = Read-JsonFile (
    Join-Path $RepositoryRoot ".github\rulesets\protected-release-tags.json"
)
$TagCreationRules = Read-JsonFile (
    Join-Path $RepositoryRoot ".github\rulesets\release-tag-creation.json"
)

if (-not $Apply) {
    Write-Host "Dry run only. No GitHub settings were changed."
    Write-Host "After the approved visibility change, run:"
    Write-Host "  .\scripts\configure-public-repository.ps1 -Apply"
    Write-Host "Prepared controls:"
    Write-Host "  - squash work merges, merge-commit promotions, and branch cleanup"
    Write-Host "  - read-only, SHA-pinned selected Actions"
    Write-Host "  - 30-day Actions log and artifact retention"
    Write-Host "  - protected main/develop branches and v* tags"
    Write-Host "  - dependency, secret, push, CodeQL, and private-reporting security"
    exit 0
}

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    throw "GitHub CLI is required."
}
& gh auth status -h github.com
if ($LASTEXITCODE -ne 0) {
    throw "GitHub CLI authentication is not valid."
}

$RepositoryState = Invoke-GitHubApi -Method GET -Endpoint "repos/$Repository" |
    ConvertFrom-Json
if ($RepositoryState.visibility -ne "public") {
    throw "The repository must be public before applying public-repository controls."
}
if ($RepositoryState.owner.id -ne $ExpectedOwnerId) {
    throw "The repository owner identity does not match the reviewed configuration."
}

Invoke-GitHubApi -Method PATCH -Endpoint "repos/$Repository" -Body @{
    allow_auto_merge = $false
    allow_merge_commit = $true
    allow_rebase_merge = $false
    allow_squash_merge = $true
    allow_update_branch = $true
    delete_branch_on_merge = $true
    description = (
        "Local-first chat, image, and video AI workspace with swappable models"
    )
    has_discussions = $false
    has_issues = $true
    has_projects = $false
    has_wiki = $false
    security_and_analysis = @{
        secret_scanning = @{ status = "enabled" }
        secret_scanning_push_protection = @{ status = "enabled" }
    }
    squash_merge_commit_message = "PR_BODY"
    squash_merge_commit_title = "PR_TITLE"
} | Out-Null

Invoke-GitHubApi -Method PUT -Endpoint "repos/$Repository/topics" -Body @{
    names = @(
        "image-generation",
        "linux",
        "llm",
        "local-ai",
        "video-generation",
        "windows"
    )
} | Out-Null

Invoke-GitHubApi -Method PUT -Endpoint "repos/$Repository/actions/permissions" -Body @{
    allowed_actions = "selected"
    enabled = $true
    sha_pinning_required = $true
} | Out-Null
Invoke-GitHubApi `
    -Method PUT `
    -Endpoint "repos/$Repository/actions/permissions/selected-actions" `
    -Body @{
        github_owned_allowed = $false
        patterns_allowed = $AllowedActionPatterns
        verified_allowed = $false
    } | Out-Null
Invoke-GitHubApi `
    -Method PUT `
    -Endpoint "repos/$Repository/actions/permissions/workflow" `
    -Body @{
        can_approve_pull_request_reviews = $false
        default_workflow_permissions = "read"
    } | Out-Null
Invoke-GitHubApi `
    -Method PUT `
    -Endpoint "repos/$Repository/actions/permissions/artifact-and-log-retention" `
    -Body @{ days = 30 } | Out-Null

Invoke-GitHubApi -Method PUT -Endpoint "repos/$Repository/vulnerability-alerts" | Out-Null
Invoke-GitHubApi `
    -Method PUT `
    -Endpoint "repos/$Repository/automated-security-fixes" | Out-Null
Invoke-GitHubApi `
    -Method PUT `
    -Endpoint "repos/$Repository/private-vulnerability-reporting" | Out-Null
Invoke-GitHubApi `
    -Method PATCH `
    -Endpoint "repos/$Repository/code-scanning/default-setup" `
    -Body @{
        languages = @("actions", "javascript-typescript", "python")
        query_suite = "extended"
        runner_type = "standard"
        state = "configured"
        threat_model = "remote_and_local"
    } | Out-Null

Set-Ruleset -Definition $BranchRules
Set-Ruleset -Definition $TagRules
Set-Ruleset -Definition $TagCreationRules

$PrivateReporting = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/private-vulnerability-reporting" |
    ConvertFrom-Json
$FinalRepository = Invoke-GitHubApi -Method GET -Endpoint "repos/$Repository" |
    ConvertFrom-Json
$FinalTopics = Invoke-GitHubApi -Method GET -Endpoint "repos/$Repository/topics" |
    ConvertFrom-Json
$ActionPermissions = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/actions/permissions" |
    ConvertFrom-Json
$SelectedActions = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/actions/permissions/selected-actions" |
    ConvertFrom-Json
$WorkflowPermissions = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/actions/permissions/workflow" |
    ConvertFrom-Json
$ArtifactRetention = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/actions/permissions/artifact-and-log-retention" |
    ConvertFrom-Json
$CodeScanning = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/code-scanning/default-setup" |
    ConvertFrom-Json
Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/vulnerability-alerts" | Out-Null
Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/automated-security-fixes" | Out-Null
$FinalRulesets = Invoke-GitHubApi -Method GET -Endpoint "repos/$Repository/rulesets" |
    ConvertFrom-Json

$BranchRuleSummary = @(
    $FinalRulesets | Where-Object name -eq $BranchRules.name
)
if ($BranchRuleSummary.Count -ne 1) {
    throw "The public branch ruleset could not be verified."
}
$TagRuleSummary = @(
    $FinalRulesets | Where-Object name -eq $TagRules.name
)
if ($TagRuleSummary.Count -ne 1) {
    throw "The release-tag ruleset could not be verified."
}
$TagCreationRuleSummary = @(
    $FinalRulesets | Where-Object name -eq $TagCreationRules.name
)
if ($TagCreationRuleSummary.Count -ne 1) {
    throw "The release-tag creation ruleset could not be verified."
}
$ActualBranchRules = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/rulesets/$($BranchRuleSummary[0].id)" |
    ConvertFrom-Json
$ActualTagRules = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/rulesets/$($TagRuleSummary[0].id)" |
    ConvertFrom-Json
$ActualTagCreationRules = Invoke-GitHubApi `
    -Method GET `
    -Endpoint "repos/$Repository/rulesets/$($TagCreationRuleSummary[0].id)" |
    ConvertFrom-Json

Assert-JsonSubset -Label "Repository settings" -Actual $FinalRepository -Expected @{
    allow_auto_merge = $false
    allow_merge_commit = $true
    allow_rebase_merge = $false
    allow_squash_merge = $true
    allow_update_branch = $true
    delete_branch_on_merge = $true
    description = (
        "Local-first chat, image, and video AI workspace with swappable models"
    )
    has_discussions = $false
    has_issues = $true
    has_projects = $false
    has_wiki = $false
    security_and_analysis = @{
        secret_scanning = @{ status = "enabled" }
        secret_scanning_push_protection = @{ status = "enabled" }
    }
}
Assert-JsonSubset -Label "Repository topics" -Actual $FinalTopics -Expected @{
    names = @(
        "image-generation",
        "linux",
        "llm",
        "local-ai",
        "video-generation",
        "windows"
    )
}
Assert-JsonSubset -Label "Actions policy" -Actual $ActionPermissions -Expected @{
    allowed_actions = "selected"
    enabled = $true
    sha_pinning_required = $true
}
Assert-JsonSubset -Label "Selected Actions policy" -Actual $SelectedActions -Expected @{
    github_owned_allowed = $false
    patterns_allowed = $AllowedActionPatterns
    verified_allowed = $false
}
Assert-JsonSubset -Label "Workflow token policy" -Actual $WorkflowPermissions -Expected @{
    can_approve_pull_request_reviews = $false
    default_workflow_permissions = "read"
}
Assert-JsonSubset -Label "Actions retention" -Actual $ArtifactRetention -Expected @{
    days = 30
}
Assert-JsonSubset -Label "Code scanning" -Actual $CodeScanning -Expected @{
    languages = @("actions", "javascript-typescript", "python")
    query_suite = "extended"
    runner_type = "standard"
    state = "configured"
    threat_model = "remote_and_local"
}
Assert-JsonSubset `
    -Label "Private vulnerability reporting" `
    -Actual $PrivateReporting `
    -Expected @{ enabled = $true }
Assert-JsonSubset `
    -Label "Branch ruleset" `
    -Actual $ActualBranchRules `
    -Expected $BranchRules
Assert-JsonSubset `
    -Label "Release-tag ruleset" `
    -Actual $ActualTagRules `
    -Expected $TagRules
Assert-JsonSubset `
    -Label "Release-tag creation ruleset" `
    -Actual $ActualTagCreationRules `
    -Expected $TagCreationRules

Write-Host "Public repository controls were applied and verified for $Repository."
