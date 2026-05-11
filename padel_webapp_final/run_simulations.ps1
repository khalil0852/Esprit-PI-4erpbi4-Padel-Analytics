# =============================================================================
# PADEL ANALYTICS — ALL ALERTS FIRING SIMULATION
# Goal: Trigger HighLatency + HighErrorRate + AccuracyDegradation + DataDrift
# =============================================================================

$BASE_URL = "http://localhost:5000"

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "  PADEL ANALYTICS - ALL ALERTS FIRING SIMULATION" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Open these BEFORE starting:" -ForegroundColor Yellow
Write-Host "  - Grafana: http://localhost:3000" -ForegroundColor Yellow
Write-Host "  - Prometheus alerts: http://localhost:9090/alerts" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to start"

# =============================================================================
# STEP 1: TRIGGER DATADRIFT (data_freshness > 24h)
# =============================================================================
Write-Host ""
Write-Host "[STEP 1] Setting data_freshness to 48 hours..." -ForegroundColor Magenta
try {
    Invoke-WebRequest -Method POST -Uri "$BASE_URL/api/simulate-drift" `
        -Body '{"accuracy": 0.854, "model": "data_test", "freshness_hours": 48}' `
        -ContentType "application/json" `
        -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | Out-Null
    Write-Host "  Done. DataDrift will fire in ~1 minute" -ForegroundColor Green
} catch {
    Write-Host "  Warning: simulate-drift endpoint may not support freshness_hours parameter" -ForegroundColor Yellow
}

# =============================================================================
# STEP 2: TRIGGER ACCURACYDEGRADATION (drift all 13 models)
# =============================================================================
Write-Host ""
Write-Host "[STEP 2] Drifting all 13 models -25% below baseline..." -ForegroundColor Magenta

$models = @(
    @{ name = "random_forest"; baseline = 0.918 },
    @{ name = "gradient_boosting"; baseline = 0.935 },
    @{ name = "xgboost_classifier"; baseline = 0.928 },
    @{ name = "ridge"; baseline = 0.812 },
    @{ name = "lasso"; baseline = 0.798 },
    @{ name = "xgboost_regressor"; baseline = 0.879 },
    @{ name = "kmeans"; baseline = 0.473 },
    @{ name = "gmm"; baseline = 0.451 },
    @{ name = "hierarchical"; baseline = 0.428 },
    @{ name = "isolation_forest"; baseline = 0.95 },
    @{ name = "lof"; baseline = 0.92 },
    @{ name = "arima"; baseline = 0.749 },
    @{ name = "prophet"; baseline = 0.754 }
)

foreach ($model in $models) {
    $new_acc = [math]::Round($model.baseline * 0.75, 4)  # -25% drop, well below 0.80
    $body = @{ accuracy = $new_acc; model = $model.name } | ConvertTo-Json
    try {
        Invoke-WebRequest -Method POST -Uri "$BASE_URL/api/simulate-drift" `
            -Body $body -ContentType "application/json" `
            -UseBasicParsing -ErrorAction Stop -TimeoutSec 5 | Out-Null
        Write-Host "  $($model.name.PadRight(22)) accuracy=$new_acc DRIFT" -ForegroundColor Red
    } catch {}
}
Write-Host "  All 13 models drifted. AccuracyDegradation will fire in ~1 minute" -ForegroundColor Green

# =============================================================================
# STEP 3: SUSTAINED TRAFFIC + ERRORS for 90 seconds
# =============================================================================
Write-Host ""
Write-Host "[STEP 3] Generating high traffic + errors for 90 seconds..." -ForegroundColor Magenta
Write-Host "  (Triggers HighLatency + HighErrorRate)" -ForegroundColor Yellow

$jobs = @()

# 8 workers spamming heavy endpoint (latency)
for ($i = 0; $i -lt 8; $i++) {
    $jobs += Start-Job -ScriptBlock {
        param($base)
        $end = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $end) {
            try {
                Invoke-WebRequest -Method POST -Uri "$base/api/predict-team" `
                    -Body '{"team_a":["Agustin Tapia","Arturo Coello"],"team_b":["Ale Galan","Federico Chingotto"],"round":"Final"}' `
                    -ContentType "application/json" `
                    -UseBasicParsing -TimeoutSec 30 -ErrorAction SilentlyContinue | Out-Null
            } catch {}
        }
    } -ArgumentList $BASE_URL
}

# 4 workers generating errors
for ($i = 0; $i -lt 4; $i++) {
    $jobs += Start-Job -ScriptBlock {
        param($base)
        $end = (Get-Date).AddSeconds(90)
        while ((Get-Date) -lt $end) {
            try { Invoke-WebRequest -Method GET -Uri "$base/api/error-$(Get-Random)" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null } catch {}
            try { Invoke-WebRequest -Method POST -Uri "$base/api/points" -Body 'malformed{{{' -ContentType "application/json" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null } catch {}
            Start-Sleep -Milliseconds 50
        }
    } -ArgumentList $BASE_URL
}

# Countdown
for ($i = 90; $i -gt 0; $i -= 10) {
    Write-Host "  $i seconds remaining..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}

$jobs | Stop-Job -ErrorAction SilentlyContinue
$jobs | Remove-Job -Force

# =============================================================================
# STEP 4: HOLD ALERTS FIRING — Take screenshots NOW
# =============================================================================
Write-Host ""
Write-Host "==================================================================" -ForegroundColor Red
Write-Host "  ALL 4 ALERTS SHOULD NOW BE FIRING" -ForegroundColor Red
Write-Host "==================================================================" -ForegroundColor Red
Write-Host ""
Write-Host "  TAKE SCREENSHOTS NOW:" -ForegroundColor Yellow
Write-Host "  1. http://localhost:9090/alerts (4 alerts firing)" -ForegroundColor Cyan
Write-Host "  2. http://localhost:3000 (Grafana panels)" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Keeping alerts alive for 60 more seconds..." -ForegroundColor Yellow

# Light background traffic to keep error rate elevated
$keepAlive = Start-Job -ScriptBlock {
    param($base)
    $end = (Get-Date).AddSeconds(60)
    while ((Get-Date) -lt $end) {
        try { Invoke-WebRequest -Method GET -Uri "$base/api/error-$(Get-Random)" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue | Out-Null } catch {}
        Start-Sleep -Milliseconds 200
    }
} -ArgumentList $BASE_URL

for ($i = 60; $i -gt 0; $i -= 10) {
    Write-Host "  $i seconds left..." -ForegroundColor Yellow
    Start-Sleep -Seconds 10
}

$keepAlive | Stop-Job -ErrorAction SilentlyContinue
$keepAlive | Remove-Job -Force

# =============================================================================
# RESET
# =============================================================================
Write-Host ""
Write-Host "[RESET] Restoring all baselines..." -ForegroundColor Green
foreach ($model in $models) {
    $body = @{ accuracy = $model.baseline; model = $model.name } | ConvertTo-Json
    try {
        Invoke-WebRequest -Method POST -Uri "$BASE_URL/api/simulate-drift" `
            -Body $body -ContentType "application/json" `
            -UseBasicParsing -ErrorAction SilentlyContinue -TimeoutSec 5 | Out-Null
    } catch {}
}

try {
    Invoke-WebRequest -Method POST -Uri "$BASE_URL/api/simulate-drift" `
        -Body '{"accuracy": 0.854, "model": "data_test", "freshness_hours": 0}' `
        -ContentType "application/json" `
        -UseBasicParsing -TimeoutSec 5 -ErrorAction SilentlyContinue | Out-Null
} catch {}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Green
Write-Host "  SIMULATION COMPLETE" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Green
Write-Host ""
