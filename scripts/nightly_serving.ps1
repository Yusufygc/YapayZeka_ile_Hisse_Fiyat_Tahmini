# E2 Faz 8 - gecelik serving pipeline sarmalayicisi (Windows Task Scheduler hedefi).
# Her gun 21:00'de calisir (BIST kapanis sonrasi): islem-gunu kapisi -> veri tazeleme -> skorlama -> PeerStore.
# Cikti logs/nightly_<yyyyMMdd>.log dosyasina (UTF-8, BOM'suz) akar; 14 gunden eski loglar budanir.
# Bitis satiri python istisna atsa / sarmalayici hata verse de try/finally ile HER ZAMAN yazilir.
# (Surec dis kaynakli olursa (Task Scheduler zaman siniri, kapanis) bitis satiri yazilamaz; bu durum
# data/nightly_status.json'da completed=false olarak gorunur.)

$ErrorActionPreference = "Stop"

# Repo koku = bu scriptin bir ust dizini (scripts/ -> repo).
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TQDM_DISABLE = "1"

# Python yorumlayici: AI_CORE_PYTHON env -> proje .venv -> PATH'teki python.
$Py = $null
if ($env:AI_CORE_PYTHON -and (Test-Path $env:AI_CORE_PYTHON)) {
    $Py = $env:AI_CORE_PYTHON
} else {
    $VenvPy = Join-Path $RepoRoot ".venv\Scripts\python.exe"
    if (Test-Path $VenvPy) {
        $Py = $VenvPy
    } else {
        $Cmd = Get-Command python -ErrorAction SilentlyContinue
        if ($Cmd) { $Py = $Cmd.Source }
    }
}
if (-not $Py) {
    [Console]::Error.WriteLine("python bulunamadi (AI_CORE_PYTHON, .venv\Scripts\python.exe, PATH denendi)")
    exit 2
}

$LogDir = Join-Path $RepoRoot "logs"
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$Stamp = Get-Date -Format "yyyyMMdd"
$Log = Join-Path $LogDir "nightly_$Stamp.log"

# Log tek yerden, UTF-8 (BOM'suz) yazilir. PS 5.1'de `*>>` / `>>` yonlendirmesi ONCEKI -Encoding'i yok sayip
# UTF-16 uretir; bu yuzden python ciktisi boru hattindan okunup StreamWriter ile yazilir.
$Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$Stream = New-Object System.IO.FileStream($Log, [System.IO.FileMode]::Append,
    [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
$Writer = New-Object System.IO.StreamWriter($Stream, $Utf8NoBom)
$Writer.AutoFlush = $true

$rc = 1
try {
    $Writer.WriteLine("==== nightly_serving baslangic $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====")

    # Python UTF-8 yazar; PS'in native cikti kod cozumu de UTF-8 olmali (konsolsuz gorevde atanamayabilir).
    try {
        [Console]::OutputEncoding = $Utf8NoBom
    } catch {
        $Writer.WriteLine("[WARN] Console.OutputEncoding ayarlanamadi: $($_.Exception.Message)")
    }

    # 2>&1 ile stderr satirlari ErrorRecord olur; "Stop" altinda ilk stderr satiri betigi keserdi -> Continue.
    $ErrorActionPreference = "Continue"
    # --model ensemble (E2 Faz 9): LGB+MLP 2-bacak ensemble; ~4x gece hesap, IC +%18.
    & $Py -m src.serving.nightly --db "data/serving_pool.db" --boost 400 --model ensemble --fallback-dir "data/yedek" 2>&1 |
        ForEach-Object {
            if ($_ -is [System.Management.Automation.ErrorRecord]) { $Writer.WriteLine($_.Exception.Message) }
            else { $Writer.WriteLine([string]$_) }
        }
    $rc = $LASTEXITCODE
} catch {
    $Writer.WriteLine("[HATA] sarmalayici istisnasi: $($_.Exception.Message)")
    $rc = 1
} finally {
    $Writer.WriteLine("==== nightly_serving bitis exit=$rc $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ====")
    $Writer.Dispose()
    $Stream.Dispose()

    # 14 gunden eski nightly loglarini buda.
    Get-ChildItem -Path $LogDir -Filter "nightly_*.log" -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-14) } |
        Remove-Item -Force -ErrorAction SilentlyContinue
}

exit $rc
