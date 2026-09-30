# Скриншот ТОЛЬКО окна по заголовку через PrintWindow (без захвата экрана:
# окно может быть перекрыто — чужой контент не попадает в кадр).
param(
    [string]$Title = "RAG view repro",
    [string]$OutPath = "C:\AI\AI_2_1\deepseek-chat-app\output\window.png"
)
Add-Type -AssemblyName System.Drawing
Add-Type @"
using System;
using System.Runtime.InteropServices;
public class Win32 {
    [DllImport("user32.dll")] public static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);
    [DllImport("user32.dll")] public static extern bool PrintWindow(IntPtr hWnd, IntPtr hdc, uint flags);
    [DllImport("user32.dll")] public static extern bool ShowWindow(IntPtr hWnd, int cmd);
    public struct RECT { public int Left, Top, Right, Bottom; }
}
"@
$proc = Get-Process | Where-Object { $_.MainWindowTitle -eq $Title } | Select-Object -First 1
if (-not $proc) { Write-Output "window not found: $Title"; exit 1 }
$hwnd = $proc.MainWindowHandle
[Win32]::ShowWindow($hwnd, 9) | Out-Null   # SW_RESTORE (свёрнутое окно)
Start-Sleep -Milliseconds 1200
$rect = New-Object Win32+RECT
[Win32]::GetWindowRect($hwnd, [ref]$rect) | Out-Null
$w = $rect.Right - $rect.Left
$h = $rect.Bottom - $rect.Top
if ($w -le 0 -or $h -le 0) { Write-Output "bad rect"; exit 1 }
$bitmap = New-Object System.Drawing.Bitmap($w, $h)
$graphics = [System.Drawing.Graphics]::FromImage($bitmap)
$hdc = $graphics.GetHdc()
# PW_RENDERFULLCONTENT (2) — рендерит содержимое перекрытого окна (DWM)
$ok = [Win32]::PrintWindow($hwnd, $hdc, 2)
$graphics.ReleaseHdc($hdc)
$graphics.Dispose()
if (-not $ok) { Write-Output "PrintWindow failed"; exit 1 }
New-Item -ItemType Directory -Force -Path (Split-Path $OutPath) | Out-Null
$bitmap.Save($OutPath, [System.Drawing.Imaging.ImageFormat]::Png)
$bitmap.Dispose()
Write-Output "saved: $OutPath (${w}x${h})"
