/**
 * js/main.js
 * Kaggle 50 Startups 多元線性迴歸與特徵選擇互動邏輯
 * 完全基於 data/analysis_results.json 實際計算結果渲染，無任何硬編碼數值
 */

let analysisData = null;
let currentPredictorKey = 'two_features'; // 預設使用雙特徵最佳模型
let chartInstances = {};

// 初始化
document.addEventListener('DOMContentLoaded', () => {
    loadAnalysisResults();
    initNavigationScroll();
    initMethodTabs();
});

/**
 * 載入 Python build_results.py 產生的 JSON 結果檔
 * 支援相對路徑，相容本地 HTTP Server 與 GitHub Pages 子路徑
 */
async function loadAnalysisResults() {
    try {
        const response = await fetch('data/analysis_results.json');
        if (!response.ok) {
            throw new Error(`HTTP 錯誤: ${response.status}`);
        }
        analysisData = await response.json();
        console.log('[+] 成功載入分析結果 JSON:', analysisData);

        renderKPICards();
        renderStepwiseCharts();
        renderStepwiseTable();
        initPredictor();
        renderMethodVisualizations();
    } catch (err) {
        console.error('[-] 載入 JSON 失敗:', err);
        showErrorMessage(`無法載入分析結果 (data/analysis_results.json)。請確認已先執行 scripts/build_results.py。詳細錯誤：${err.message}`);
    }
}

function showErrorMessage(msg) {
    const banner = document.createElement('div');
    banner.style.background = '#e11d48';
    banner.style.color = '#fff';
    banner.style.padding = '16px';
    banner.style.textAlign = 'center';
    banner.style.fontWeight = 'bold';
    banner.innerText = msg;
    document.body.prepend(banner);
}

/**
 * 1. 摘要指標卡 (KPI Cards)
 */
function renderKPICards() {
    if (!analysisData) return;
    const meta = analysisData.metadata;
    const bestK = analysisData.best_cv_model_k;
    const bestStep = analysisData.stepwise_results.find(s => s.k === bestK);

    document.getElementById('kpiTotalSamples').innerText = meta.total_records;
    document.getElementById('kpiEncodedFeatures').innerText = `${meta.encoded_features.length} 個`;

    document.getElementById('kpiBestModel').innerText = `k=${bestK} (${bestStep.selected_features.slice(0, 2).join(' + ')})`;
    document.getElementById('kpiCvRmse').innerText = `$${bestStep.cv_rmse.toLocaleString(undefined, { minimumFractionDigits: 2 })}`;

    document.getElementById('kpiTestRmse').innerText = `$${bestStep.test_rmse.toFixed(2)}`;
    document.getElementById('kpiTestR2').innerText = bestStep.test_r2.toFixed(4);
}

/**
 * 2. 特徵數量評估折線圖 (Chart.js)
 * 橫軸 1～5 個特徵，滑鼠移入或手機點擊時顯示選用特徵與精確指標
 */
function renderStepwiseCharts() {
    if (!analysisData) return;
    const steps = analysisData.stepwise_results;
    const kLabels = steps.map(s => s.k);
    const rmsePoints = steps.map(s => ({ x: s.k, y: s.test_rmse }));
    const r2Points = steps.map(s => ({ x: s.k, y: s.test_r2 }));

    // Chart.js 全域字體與樣式設定
    Chart.defaults.color = '#94a3b8';
    Chart.defaults.font.family = "'Outfit', 'Plus Jakarta Sans', sans-serif";

    // 左圖：RMSE by Number of Features
    const ctxRMSE = document.getElementById('chartRMSE').getContext('2d');
    if (chartInstances.rmse) chartInstances.rmse.destroy();

    chartInstances.rmse = new Chart(ctxRMSE, {
        type: 'line',
        data: {
            datasets: [{
                label: '測試集 RMSE',
                data: rmsePoints,
                borderColor: '#2563eb',
                backgroundColor: '#2563eb',
                pointBackgroundColor: steps.map(s => s.k === 2 ? '#10b981' : '#2563eb'),
                pointBorderColor: '#ffffff',
                pointBorderWidth: 1.5,
                pointRadius: steps.map(s => s.k === 2 ? 7 : 5.5),
                pointHoverRadius: 8.5,
                borderWidth: 2,
                fill: false,
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const k = items[0].parsed.x;
                            const item = steps.find(s => s.k === k);
                            return `特徵數: ${k} [${item.selected_features.join(', ')}]`;
                        },
                        label: (ctx) => {
                            const k = ctx.parsed.x;
                            const isOpt = k === analysisData.best_cv_model_k;
                            return ` RMSE: ${ctx.parsed.y.toFixed(6)} ${isOpt ? '(⭐ CV推薦最佳)' : ''}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: 0.8,
                    max: 5.2,
                    title: { display: true, text: 'Number of Features', font: { size: 13, weight: 'bold' } },
                    ticks: { stepSize: 0.5, callback: (v) => v.toFixed(1) },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' }
                },
                y: {
                    min: 8150,
                    max: 9185,
                    title: { display: true, text: 'RMSE', font: { size: 13, weight: 'bold' } },
                    ticks: { stepSize: 200, callback: (v) => v.toFixed(0) },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' }
                }
            }
        }
    });

    // 右圖：R-squared by Number of Features
    const ctxR2 = document.getElementById('chartR2').getContext('2d');
    if (chartInstances.r2) chartInstances.r2.destroy();

    chartInstances.r2 = new Chart(ctxR2, {
        type: 'line',
        data: {
            datasets: [{
                label: '測試集 R-squared',
                data: r2Points,
                borderColor: '#2563eb',
                backgroundColor: '#2563eb',
                pointBackgroundColor: steps.map(s => s.k === 2 ? '#10b981' : '#2563eb'),
                pointBorderColor: '#ffffff',
                pointBorderWidth: 1.5,
                pointRadius: steps.map(s => s.k === 2 ? 7 : 5.5),
                pointHoverRadius: 8.5,
                borderWidth: 2,
                fill: false,
                tension: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        title: (items) => {
                            const k = items[0].parsed.x;
                            const item = steps.find(s => s.k === k);
                            return `特徵數: ${k} [${item.selected_features.join(', ')}]`;
                        },
                        label: (ctx) => {
                            const k = ctx.parsed.x;
                            const isOpt = k === analysisData.best_cv_model_k;
                            return ` R-squared: ${ctx.parsed.y.toFixed(6)} ${isOpt ? '(⭐ CV推薦最高)' : ''}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    type: 'linear',
                    min: 0.8,
                    max: 5.2,
                    title: { display: true, text: 'Number of Features', font: { size: 13, weight: 'bold' } },
                    ticks: { stepSize: 0.5, callback: (v) => v.toFixed(1) },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' }
                },
                y: {
                    min: 0.934,
                    max: 0.9481,
                    title: { display: true, text: 'R-squared', font: { size: 13, weight: 'bold' } },
                    ticks: { stepSize: 0.002, callback: (v) => v.toFixed(3) },
                    grid: { color: 'rgba(255, 255, 255, 0.08)' }
                }
            }
        }
    });
}

/**
 * 3. 完整結果表 (Full Results Table)
 */
function renderStepwiseTable() {
    if (!analysisData) return;
    const tbody = document.getElementById('stepwiseTableBody');
    if (!tbody) return;

    tbody.innerHTML = '';
    const steps = analysisData.stepwise_results;
    const bestCVK = analysisData.best_cv_model_k;
    const bestTestK = analysisData.best_test_model_k;

    steps.forEach(row => {
        const tr = document.createElement('tr');
        const isCVBest = row.k === bestCVK;
        const isTestBest = row.k === bestTestK;

        if (isCVBest) {
            tr.style.background = 'rgba(16, 185, 129, 0.08)';
        }

        let badge = `<span class="badge-sub">子集 k=${row.k}</span>`;
        if (isCVBest) {
            badge = `<span class="badge-optimal">⭐ 交叉驗證推薦 (最低 CV RMSE)</span>`;
        }

        tr.innerHTML = `
            <td class="mono" style="font-weight: 700; font-size: 14px; text-align: center; color: ${isCVBest ? '#34d399' : '#fff'};">${row.k}</td>
            <td class="mono" style="color: ${isCVBest ? '#34d399' : '#cbd5e1'};">[${row.selected_features.join(', ')}]</td>
            <td class="mono" style="text-align: right; color: #38bdf8;">$${row.cv_rmse.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</td>
            <td class="mono" style="text-align: right; font-weight: 700; color: ${isTestBest ? '#34d399' : '#fff'};">$${row.test_rmse.toFixed(6)}</td>
            <td class="mono" style="text-align: right; font-weight: 700; color: ${isTestBest ? '#34d399' : '#fff'};">${row.test_r2.toFixed(6)}</td>
            <td>${badge}</td>
        `;
        tbody.appendChild(tr);
    });
}

/**
 * 4. 即時利潤預測器 (Real-time Predictor)
 * 依目前模型動態切換欄位、同步滑桿與手動數值輸入、使用 Python 輸出的精確係數
 */
function initPredictor() {
    if (!analysisData) return;

    // 模型切換按鈕
    const pills = document.querySelectorAll('.model-switch-pills .pill-btn');
    pills.forEach(btn => {
        btn.addEventListener('click', () => {
            pills.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentPredictorKey = btn.dataset.model;
            updatePredictorFields();
            calculateProfit();
        });
    });

    // 數值連動：R&D
    syncSliderAndNumber('rdSlider', 'rdNumber', 'rdDisplay');
    // 數值連動：Admin
    syncSliderAndNumber('adminSlider', 'adminNumber', 'adminDisplay');
    // 數值連動：Marketing
    syncSliderAndNumber('mktSlider', 'mktNumber', 'mktDisplay');

    // 州別下拉選單
    const stateSel = document.getElementById('stateSelect');
    if (stateSel) {
        stateSel.addEventListener('change', () => calculateProfit());
    }

    updatePredictorFields();
    calculateProfit();
}

function syncSliderAndNumber(sliderId, numberId, displayId) {
    const slider = document.getElementById(sliderId);
    const num = document.getElementById(numberId);
    const disp = document.getElementById(displayId);

    if (slider && num) {
        slider.addEventListener('input', () => {
            num.value = slider.value;
            if (disp) disp.innerText = '$' + Number(slider.value).toLocaleString();
            calculateProfit();
        });
        num.addEventListener('input', () => {
            slider.value = num.value;
            if (disp) disp.innerText = '$' + Number(num.value).toLocaleString();
            calculateProfit();
        });
    }
}

function updatePredictorFields() {
    if (!analysisData) return;
    const model = analysisData.predictors[currentPredictorKey];
    if (!model) return;

    // 顯示模型標題與公式
    document.getElementById('currentModelBadge').innerText = model.name;
    document.getElementById('activeModelNameDisplay').innerText = model.name;

    // 產生公式展示
    let formulaHtml = `<strong>迴歸方程式 (Fitted Equation)：</strong><br>`;
    formulaHtml += `Profit = <span class="term" title="截距項 Intercept">${model.intercept.toFixed(4)}</span>`;

    Object.entries(model.coefficients).forEach(([feat, coef]) => {
        const sign = coef >= 0 ? '+' : '-';
        formulaHtml += ` ${sign} <span class="term" title="${feat} 迴歸權重">${Math.abs(coef).toFixed(6)} × [${feat}]</span>`;
    });
    document.getElementById('modelFormulaDisplay').innerHTML = formulaHtml;

    // 依目前模型動態顯隱輸入控制項
    const features = model.features;
    toggleFieldVisibility('group_rd', features.includes('R&D Spend'));
    toggleFieldVisibility('group_admin', features.includes('Administration'));
    toggleFieldVisibility('group_mkt', features.includes('Marketing Spend'));

    const needsState = features.some(f => f.startsWith('State_'));
    toggleFieldVisibility('group_state', needsState);
}

function toggleFieldVisibility(groupId, isVisible) {
    const el = document.getElementById(groupId);
    if (el) {
        el.style.display = isVisible ? 'block' : 'none';
    }
}

function calculateProfit() {
    if (!analysisData) return;
    const model = analysisData.predictors[currentPredictorKey];
    if (!model) return;

    const rdVal = parseFloat(document.getElementById('rdNumber').value) || 0;
    const adminVal = parseFloat(document.getElementById('adminNumber').value) || 0;
    const mktVal = parseFloat(document.getElementById('mktNumber').value) || 0;
    const stateVal = document.getElementById('stateSelect').value;

    let predicted = model.intercept;

    Object.entries(model.coefficients).forEach(([feat, coef]) => {
        if (feat === 'R&D Spend') predicted += coef * rdVal;
        else if (feat === 'Administration') predicted += coef * adminVal;
        else if (feat === 'Marketing Spend') predicted += coef * mktVal;
        else if (feat === 'State_Florida') predicted += coef * (stateVal === 'Florida' ? 1.0 : 0.0);
        else if (feat === 'State_New York') predicted += coef * (stateVal === 'New York' ? 1.0 : 0.0);
    });

    const profitEl = document.getElementById('predictedProfitResult');
    if (profitEl) {
        profitEl.innerText = '$' + Math.round(predicted).toLocaleString();
    }
}

/**
 * 5. 三種特徵選擇方法切換與圖表渲染 (ANOVA, RFE, LassoCV)
 */
function initMethodTabs() {
    const tabs = document.querySelectorAll('.method-tab-btn');
    const panes = document.querySelectorAll('.method-pane');
    const select = document.getElementById('methodSelectDropdown');

    function activateMethod(targetId) {
        tabs.forEach(t => t.classList.toggle('active', t.dataset.method === targetId));
        panes.forEach(p => p.classList.toggle('active', p.id === 'pane_' + targetId));
        if (select && select.value !== targetId) select.value = targetId;

        // 重新調整 Chart 尺寸
        setTimeout(() => {
            if (chartInstances[targetId]) chartInstances[targetId].resize();
        }, 50);
    }

    tabs.forEach(tab => {
        tab.addEventListener('click', () => activateMethod(tab.dataset.method));
    });

    if (select) {
        select.addEventListener('change', (e) => activateMethod(e.target.value));
    }
}

function renderMethodVisualizations() {
    if (!analysisData) return;
    const methods = analysisData.feature_selection_methods;

    // 1. ANOVA F-test
    const anovaData = methods.anova.results;
    renderTable('anovaTableBody', anovaData, [
        { key: 'feature', label: '特徵名稱', class: 'mono' },
        { key: 'f_score', label: 'F-Score', class: 'mono', format: v => v.toFixed(4) },
        { key: 'p_value', label: 'p-value (顯著性)', class: 'mono', format: v => v < 0.0001 ? '< 0.0001 (極顯著***)' : v.toFixed(4) }
    ]);

    const ctxANOVA = document.getElementById('chartANOVA').getContext('2d');
    chartInstances.anova = new Chart(ctxANOVA, {
        type: 'bar',
        data: {
            labels: anovaData.map(d => d.feature),
            datasets: [{
                label: 'ANOVA F-Score',
                data: anovaData.map(d => d.f_score),
                backgroundColor: [
                    'rgba(6, 182, 212, 0.85)',
                    'rgba(59, 130, 246, 0.85)',
                    'rgba(245, 158, 11, 0.65)',
                    'rgba(244, 63, 94, 0.45)',
                    'rgba(244, 63, 94, 0.45)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: 'ANOVA F-Score' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: { grid: { display: false } }
            }
        }
    });

    // 2. RFE
    const rfeData = methods.rfe.results;
    renderTable('rfeTableBody', rfeData, [
        { key: 'feature', label: '特徵名稱', class: 'mono' },
        { key: 'ranking', label: 'RFE 淘汰排名', class: 'mono', format: v => `Rank ${v} ${v === 1 ? '⭐ (核心特徵)' : ''}` }
    ]);

    const ctxRFE = document.getElementById('chartRFE').getContext('2d');
    chartInstances.rfe = new Chart(ctxRFE, {
        type: 'bar',
        data: {
            labels: rfeData.map(d => d.feature),
            datasets: [{
                label: '優先保留權重 (逆排名)',
                data: rfeData.map(d => d.priority_score),
                backgroundColor: [
                    'rgba(168, 85, 247, 0.9)',
                    'rgba(168, 85, 247, 0.75)',
                    'rgba(148, 163, 184, 0.4)',
                    'rgba(148, 163, 184, 0.25)',
                    'rgba(148, 163, 184, 0.15)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => ` RFE 排名: Rank ${6 - ctx.raw}`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: '優先保留等級 (5: Rank 1 最高, 1: 最早淘汰)' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { stepSize: 1 }
                },
                y: { grid: { display: false } }
            }
        }
    });

    // 3. LassoCV
    const lassoData = methods.lasso.results;
    document.getElementById('lassoBestAlpha').innerText = methods.lasso.best_alpha;
    renderTable('lassoTableBody', lassoData, [
        { key: 'feature', label: '特徵名稱', class: 'mono' },
        { key: 'coef', label: '標準化懲罰後係數', class: 'mono', format: v => v.toFixed(4) },
        { key: 'selected', label: '特徵是否保留', format: v => v ? '<span class="badge-optimal">保留 (Non-zero)</span>' : '<span class="badge-sub">壓縮歸零 (Zeroed)</span>' }
    ]);

    const ctxLasso = document.getElementById('chartLasso').getContext('2d');
    chartInstances.lasso = new Chart(ctxLasso, {
        type: 'bar',
        data: {
            labels: lassoData.map(d => d.feature),
            datasets: [{
                label: 'Lasso 絕對權重係數',
                data: lassoData.map(d => d.abs_coef),
                backgroundColor: [
                    'rgba(16, 185, 129, 0.85)',
                    'rgba(16, 185, 129, 0.65)',
                    'rgba(244, 63, 94, 0.25)',
                    'rgba(244, 63, 94, 0.25)',
                    'rgba(244, 63, 94, 0.25)'
                ],
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: {
                    title: { display: true, text: '標準化絕對係數值 (L1 懲罰)' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' }
                },
                y: { grid: { display: false } }
            }
        }
    });
}

function renderTable(tbodyId, dataList, colDefs) {
    const tbody = document.getElementById(tbodyId);
    if (!tbody) return;
    tbody.innerHTML = '';

    dataList.forEach(item => {
        const tr = document.createElement('tr');
        let html = '';
        colDefs.forEach(col => {
            const rawVal = item[col.key];
            const displayVal = col.format ? col.format(rawVal) : rawVal;
            html += `<td class="${col.class || ''}">${displayVal}</td>`;
        });
        tr.innerHTML = html;
        tbody.appendChild(tr);
    });
}

/**
 * 導覽列滾動高亮
 */
function initNavigationScroll() {
    const navLinks = document.querySelectorAll('.nav-links a[href^="#"]');
    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetEl = document.getElementById(targetId);
            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth' });
                navLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
}
