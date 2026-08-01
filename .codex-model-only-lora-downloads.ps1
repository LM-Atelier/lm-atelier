param([Parameter(Mandatory = $true)][string]$TargetPath)

$encoding = [System.Text.UTF8Encoding]::new($false)
$content = [System.IO.File]::ReadAllText($TargetPath, $encoding)

if (-not $content.Contains('    checkpoint_lora_extension,')) {
    throw "Expected auxiliary-assets import was not found."
}
$content = $content.Replace('    checkpoint_lora_extension,', '    detect_lora_extension,')

if (-not $content.Contains('lora_extension = checkpoint_lora_extension(compiled.api_graph)')) {
    throw "Expected template LoRA detector call was not found."
}
$content = $content.Replace(
    'lora_extension = checkpoint_lora_extension(compiled.api_graph)',
    'lora_extension = detect_lora_extension(compiled.api_graph)'
)

[System.IO.File]::WriteAllText($TargetPath, $content, $encoding)
