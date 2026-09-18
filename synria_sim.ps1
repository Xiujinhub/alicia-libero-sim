<#
.SYNOPSIS
    用 MuJoCo 打开 Synria 机器人模型（Alicia / Bessica / Corina）。

.DESCRIPTION
    模型取自本机下载的 Synria-Robot-Descriptions-main 仓库中已经自带的 MJCF 文件：
        <MuJoCo 根目录>\Synria-Robot-Descriptions-main\synriard\mjcf\**\*.xml
    这些 XML 的 <compiler meshdir> 都是相对路径，所以无需任何转换即可被 MuJoCo 直接加载。

.PARAMETER Model
    模型名（支持模糊匹配，例如 Alicia、Alicia_D_v5_6、gripper_50mm）。
    省略时打开默认模型 Alicia_D_v5_6_gripper_100mm。

.PARAMETER Viewer
    simulate = bin\simulate.exe            MuJoCo 官方模拟器（默认，双击式 GUI）
    studio   = bin\mujoco_studio.exe       MuJoCo Studio（新版 GUI，带可视化编辑器）
    python   = conda 环境里的 python -m mujoco.viewer（Python 可视化器）

.PARAMETER EnvName
    -Viewer python 时使用的 conda 环境名，默认 lerobot。

.PARAMETER List
    只列出所有可用模型，不启动任何程序。

.EXAMPLE
    .\synria_sim.ps1 -List
.EXAMPLE
    .\synria_sim.ps1
    # 打开 Alicia_D_v5_6_gripper_100mm.xml
.EXAMPLE
    .\synria_sim.ps1 -Model Alicia_M_v1_2_follower
    # 打开 Alicia_M_v1_2_follower.xml
.EXAMPLE
    .\synria_sim.ps1 -Model Alicia_D_v5_6_gripper_50mm -Viewer python
#>
[CmdletBinding()]
param(
    [string]$Model = '',
    [ValidateSet('simulate', 'studio', 'python')]
    [string]$Viewer = 'simulate',
    [string]$EnvName = 'lerobot',
    [switch]$List
)

$ErrorActionPreference = 'Stop'

$MujocoRoot   = $PSScriptRoot
$MjcfRoot     = Join-Path $MujocoRoot 'Synria-Robot-Descriptions-main\synriard\mjcf'
$DefaultModel = 'Alicia_D_v5_6_gripper_100mm'

if (-not (Test-Path -LiteralPath $MjcfRoot)) {
    throw "找不到 MJCF 目录：$MjcfRoot"
}

$allXml = @(Get-ChildItem -LiteralPath $MjcfRoot -Recurse -Filter '*.xml' | Sort-Object FullName)

if ($List) {
    Write-Host "可用的 MuJoCo (MJCF) 模型，共 $($allXml.Count) 个：" -ForegroundColor Cyan
    $allXml | ForEach-Object {
        '{0,-20} {1}' -f (Split-Path $_.DirectoryName -Leaf), $_.BaseName
    }
    return
}

if ([string]::IsNullOrWhiteSpace($Model)) { $Model = $DefaultModel }

$pattern = if ($Model -like '*.xml') { $Model } else { "*$Model*.xml" }
$hits    = @($allXml | Where-Object { $_.Name -like $pattern })

if ($hits.Count -gt 1) {
    # 优先精确匹配文件名 / 去扩展名后的名字
    $exact = @($hits | Where-Object { $_.Name -eq $Model -or $_.BaseName -eq $Model })
    if ($exact.Count -eq 1) { $hits = $exact }
}

if ($hits.Count -eq 0) {
    Write-Host "没有匹配 '$Model' 的模型。可用模型：" -ForegroundColor Yellow
    $allXml | ForEach-Object { '  ' + $_.BaseName }
    return
}
if ($hits.Count -gt 1) {
    Write-Host "'$Model' 匹配到多个模型，请写得更具体：" -ForegroundColor Yellow
    $hits | ForEach-Object { '  ' + $_.BaseName }
    return
}

$xmlPath = $hits[0].FullName
$xmlDir  = $hits[0].DirectoryName

switch ($Viewer) {
    'simulate' {
        $exe = Join-Path $MujocoRoot 'bin\simulate.exe'
        if (-not (Test-Path -LiteralPath $exe)) { throw "找不到 $exe" }
        Start-Process -FilePath $exe -ArgumentList "`"$xmlPath`"" -WorkingDirectory $xmlDir | Out-Null
    }
    'studio' {
        $exe = Join-Path $MujocoRoot 'bin\mujoco_studio.exe'
        if (-not (Test-Path -LiteralPath $exe)) { throw "找不到 $exe" }
        Start-Process -FilePath $exe -ArgumentList "--model_file=`"$xmlPath`"" -WorkingDirectory $xmlDir | Out-Null
    }
    'python' {
        $condaExe = if ($env:CONDA_EXE) { $env:CONDA_EXE } else { 'conda' }
        $args = @('run', '--no-capture-output', '-n', $EnvName, 'python', '-m', 'mujoco.viewer', "--mjcf=$xmlPath")
        Start-Process -FilePath $condaExe -ArgumentList $args -WorkingDirectory $xmlDir | Out-Null
    }
}

Write-Host "已用 [$Viewer] 打开：$xmlPath" -ForegroundColor Green
if ($Viewer -eq 'python') { Write-Host "conda 环境：$EnvName" -ForegroundColor Green }
