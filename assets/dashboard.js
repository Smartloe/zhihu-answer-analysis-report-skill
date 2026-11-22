/* Zhihu Answer Report dashboard runtime.
 * Reads the embedded JSON payload and renders ECharts panels.
 * Kept framework-free so the generated HTML stays self-contained.
 */
(function () {
  "use strict";

  var payloadEl = document.getElementById("report-data");
  if (!payloadEl) {
    return;
  }

  var data = {};
  try {
    data = JSON.parse(payloadEl.textContent || "{}");
  } catch (error) {
    data = {};
  }

  var root = document.documentElement;
  var themeToggle = document.getElementById("theme-toggle");
  var media = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  var storageKey = "zhihu-report-theme";
  var charts = [];
  var reducedMotion = window.matchMedia
    ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
    : false;

  function readStoredTheme() {
    try {
      var value = window.localStorage.getItem(storageKey);
      return value === "dark" || value === "light" ? value : null;
    } catch (error) {
      return null;
    }
  }

  function persistTheme(theme) {
    try {
      window.localStorage.setItem(storageKey, theme);
    } catch (error) {
      /* Storage may be unavailable in sandboxed viewers. */
    }
  }

  function systemTheme() {
    return media && media.matches ? "dark" : "light";
  }

  function applyTheme(theme, persist) {
    root.setAttribute("data-theme", theme);
    if (persist) {
      persistTheme(theme);
    }
    if (themeToggle) {
      themeToggle.textContent = theme === "dark" ? "☀" : "☾";
      themeToggle.setAttribute(
        "aria-label",
        theme === "dark" ? "切换到浅色主题" : "切换到深色主题"
      );
      themeToggle.setAttribute("title", theme === "dark" ? "浅色" : "深色");
    }
  }

  function palette() {
    var styles = window.getComputedStyle(root);
    function token(name, fallback) {
      var value = styles.getPropertyValue(name).trim();
      return value || fallback;
    }
    return {
      ink: token("--ink", "#1a1611"),
      inkSoft: token("--ink-soft", "#453d31"),
      muted: token("--ink-muted", "#7b6f5d"),
      rule: token("--rule-strong", "rgba(26,22,17,0.55)"),
      grid: token("--chart-grid", "rgba(26,22,17,0.1)"),
      surface: token("--chart-surface", "#fbf7ee"),
      accent: token("--accent", "#b8402a"),
      pine: token("--pine", "#1f5a4a"),
      indigo: token("--indigo", "#35597c"),
      ochre: token("--ochre", "#a9761c"),
      positive: token("--positive", "#1f5a4a"),
      neutral: token("--neutral", "#9a7b3c"),
      negative: token("--negative", "#a63d2f"),
      tooltipBg: token("--tooltip-bg", "rgba(26,22,17,0.95)"),
      tooltipInk: token("--tooltip-ink", "#fbf7ee")
    };
  }

  var seriesPalette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"];

  function baseText(c) {
    return {
      color: c.muted,
      fontFamily:
        '"PingFang SC", "Hiragino Sans GB", "Noto Sans SC", "Microsoft YaHei", sans-serif',
      fontSize: 11
    };
  }

  function escapeHtml(value) {
    return String(value === null || value === undefined ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function tooltipStyle(c) {
    return {
      backgroundColor: c.tooltipBg,
      borderWidth: 0,
      padding: [9, 12],
      textStyle: { color: c.tooltipInk, fontSize: 12 },
      extraCssText: "border-radius:3px;box-shadow:0 10px 28px rgba(0,0,0,.18);max-width:320px;"
    };
  }

  function axisCommon(c) {
    return {
      axisLine: { lineStyle: { color: c.rule } },
      axisTick: { show: false },
      axisLabel: Object.assign({ margin: 10 }, baseText(c)),
      splitLine: { lineStyle: { color: c.grid, type: "solid" } }
    };
  }

  function initChart(id) {
    var el = document.getElementById(id);
    if (!el || !window.echarts) {
      return null;
    }
    if (el.getAttribute("data-empty") === "1") {
      return null;
    }
    var chart = window.echarts.init(el, null, { renderer: "canvas" });
    charts.push(chart);
    return chart;
  }

  function emptyNotice() {
    var message =
      '<div class="empty-state"><strong>图表库未能加载。</strong> 请确认 assets/echarts.min.js 存在，或网络可用后刷新页面；报告中的「数据表」与 Markdown 表格仍可阅读。</div>';
    document.querySelectorAll(".chart").forEach(function (el) {
      if (el.getAttribute("data-empty") !== "1") {
        el.innerHTML = message;
      }
    });
    charts.forEach(function (chart) {
      if (chart && !chart.isDisposed()) {
        chart.dispose();
      }
    });
    charts = [];
  }

  function buildKeywords(c) {
    var chart = initChart("keywords");
    if (!chart) {
      return;
    }
    var rows = (data.topKeywords || []).slice(0, 20).slice().reverse();
    var values = rows.map(function (item) {
      return item.count;
    });
    var max = Math.max.apply(null, values.concat([1]));
    chart.setOption({
      animationDuration: reducedMotion ? 0 : 700,
      animationEasing: "cubicOut",
      grid: { left: 8, right: 46, top: 8, bottom: 8, containLabel: true },
      tooltip: Object.assign(
        {
          trigger: "axis",
          axisPointer: { type: "shadow", shadowStyle: { color: c.grid } },
          formatter: function (params) {
            var p = params[0];
            return "<b>" + escapeHtml(p.name) + "</b><br>出现 " + escapeHtml(p.value) + " 次";
          }
        },
        tooltipStyle(c)
      ),
      xAxis: Object.assign(
        {
          type: "value",
          max: Math.ceil(max * 1.12),
          axisLabel: Object.assign({}, baseText(c), { formatter: "{value}" })
        },
        {
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: c.grid } }
        }
      ),
      yAxis: Object.assign(
        {
          type: "category",
          data: rows.map(function (item) {
            return item.word;
          }),
          axisLabel: Object.assign({}, baseText(c), { fontSize: 12, color: c.inkSoft })
        },
        axisCommon(c)
      ),
      series: [
        {
          type: "bar",
          data: values,
          barWidth: 13,
          itemStyle: { color: c.accent, borderRadius: [0, 3, 3, 0] },
          emphasis: { itemStyle: { color: c.ink } },
          label: {
            show: true,
            position: "right",
            distance: 8,
            color: c.muted,
            fontSize: 11,
            fontFamily: baseText(c).fontFamily
          }
        }
      ]
    });
  }

  function sentimentColor(index, total, c) {
    if (index === total - 1) {
      return { color: c.muted, opacity: 0.55 };
    }
    var ordered = [c.negative, c.negative, c.neutral, c.positive, c.positive];
    return ordered[Math.min(index, ordered.length - 1)];
  }

  function buildSentiment(c) {
    var chart = initChart("sentiment");
    if (!chart) {
      return;
    }
    var order = data.bucketOrder || [];
    var buckets = data.sentimentBuckets || {};
    var total = order.length;
    var values = order.map(function (name) {
      return buckets[name] || 0;
    });
    var maxValue = Math.max.apply(null, values.concat([1]));
    chart.setOption({
      animationDuration: reducedMotion ? 0 : 700,
      animationEasing: "cubicOut",
      grid: { left: 8, right: 8, top: 24, bottom: 8, containLabel: true },
      tooltip: Object.assign(
        {
          trigger: "axis",
          axisPointer: { type: "shadow", shadowStyle: { color: c.grid } },
          formatter: function (params) {
            var p = params[0];
            return "<b>" + escapeHtml(p.name) + "</b><br>回答数 " + escapeHtml(p.value);
          }
        },
        tooltipStyle(c)
      ),
      xAxis: Object.assign(
        {
          type: "category",
          data: order.map(function (name) {
            return name === "unknown" ? "未知" : name.replace("-", "–");
          }),
          axisLabel: Object.assign({}, baseText(c), { fontSize: 11 })
        },
        axisCommon(c)
      ),
      yAxis: Object.assign(
        {
          type: "value",
          max: Math.ceil(maxValue * 1.18),
          axisLabel: baseText(c)
        },
        {
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: c.grid } }
        }
      ),
      series: [
        {
          type: "bar",
          data: values.map(function (value, index) {
            var color = sentimentColor(index, total, c);
            return {
              value: value,
              itemStyle: {
                color: typeof color === "string" ? color : color.color,
                opacity: typeof color === "string" ? 1 : color.opacity,
                borderRadius: [3, 3, 0, 0]
              }
            };
          }),
          barMaxWidth: 42,
          label: {
            show: true,
            position: "top",
            color: c.muted,
            fontSize: 11,
            fontFamily: baseText(c).fontFamily
          }
        }
      ]
    });
  }

  function buildTimeline(c) {
    var chart = initChart("timeline");
    if (!chart) {
      return;
    }
    var timeline = data.timeline || [];
    var values = timeline.map(function (item) {
      return item.count;
    });
    var maxValue = Math.max.apply(null, values.concat([1]));
    chart.setOption({
      animationDuration: reducedMotion ? 0 : 700,
      animationEasing: "cubicOut",
      grid: { left: 8, right: 12, top: 22, bottom: 8, containLabel: true },
      tooltip: Object.assign(
        {
          trigger: "axis",
          axisPointer: { type: "shadow", shadowStyle: { color: c.grid } },
          formatter: function (params) {
            var p = params[0];
            return "<b>" + escapeHtml(p.name) + "</b><br>回答数 " + escapeHtml(p.value);
          }
        },
        tooltipStyle(c)
      ),
      xAxis: Object.assign(
        {
          type: "category",
          data: timeline.map(function (item) {
            return item.date;
          }),
          axisLabel: Object.assign({}, baseText(c), { fontSize: 11, hideOverlap: true })
        },
        axisCommon(c)
      ),
      yAxis: Object.assign(
        {
          type: "value",
          max: Math.ceil(maxValue * 1.2),
          axisLabel: baseText(c)
        },
        {
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: c.grid } }
        }
      ),
      series: [
        {
          type: "bar",
          data: values,
          barMaxWidth: 36,
          itemStyle: { color: c.indigo, borderRadius: [3, 3, 0, 0] },
          label: {
            show: true,
            position: "top",
            color: c.muted,
            fontSize: 11,
            fontFamily: baseText(c).fontFamily
          }
        }
      ]
    });
  }

  function buildScatter(c) {
    var chart = initChart("scatter");
    if (!chart) {
      return;
    }
    var points = data.scatter || [];
    chart.setOption({
      animationDuration: reducedMotion ? 0 : 700,
      animationEasing: "cubicOut",
      grid: { left: 10, right: 24, top: 20, bottom: 74, containLabel: true },
      tooltip: Object.assign(
        {
          formatter: function (params) {
            var point = params.data;
            return (
              "<b>" +
              escapeHtml(point[3] || "未命名回答") +
              "</b><br>" +
              (point[4] ? escapeHtml(point[4]) + "<br>" : "") +
              "字数 " +
              escapeHtml(point[0]) +
              " · 赞同 " +
              escapeHtml(point[1]) +
              " · 情感 " +
              escapeHtml(point[2])
            );
          }
        },
        tooltipStyle(c)
      ),
      xAxis: Object.assign(
        {
          type: "value",
          name: "字数",
          nameLocation: "middle",
          nameGap: 28,
          nameTextStyle: baseText(c),
          axisLabel: baseText(c)
        },
        {
          axisLine: { lineStyle: { color: c.rule } },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: c.grid } }
        }
      ),
      yAxis: Object.assign(
        {
          type: "value",
          name: "赞同数",
          nameLocation: "middle",
          nameGap: 40,
          nameTextStyle: baseText(c),
          axisLabel: baseText(c)
        },
        {
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: c.grid } }
        }
      ),
      visualMap: {
        min: 0,
        max: 1,
        dimension: 2,
        orient: "horizontal",
        left: "center",
        bottom: 0,
        itemWidth: 12,
        itemHeight: 120,
        text: ["正向", "负向"],
        textStyle: baseText(c),
        inRange: { color: [c.negative, c.neutral, c.positive] },
        calculable: false
      },
      series: [
        {
          type: "scatter",
          symbolSize: 15,
          data: points,
          itemStyle: {
            borderColor: c.surface,
            borderWidth: 2,
            opacity: 0.92
          },
          emphasis: {
            scale: 1.35,
            itemStyle: { borderColor: c.ink, borderWidth: 2 }
          }
        }
      ]
    });
  }

  function buildTopics(c) {
    var chart = initChart("topics");
    if (!chart) {
      return;
    }
    var rows = (data.ldaDistribution || []).slice().sort(function (a, b) {
      return a.count - b.count;
    });
    var values = rows.map(function (item) {
      return item.count;
    });
    var maxValue = Math.max.apply(null, values.concat([1]));
    chart.setOption({
      animationDuration: reducedMotion ? 0 : 700,
      animationEasing: "cubicOut",
      grid: { left: 8, right: 52, top: 8, bottom: 8, containLabel: true },
      tooltip: Object.assign(
        {
          trigger: "axis",
          axisPointer: { type: "shadow", shadowStyle: { color: c.grid } },
          formatter: function (params) {
            var p = params[0];
            var row = rows[p.dataIndex] || {};
            return (
              "<b>" +
              escapeHtml(p.name) +
              "</b><br>回答数 " +
              escapeHtml(p.value) +
              (row.keywords
                ? "<br>关键词 " + escapeHtml(row.keywords.split(" ").join(" · "))
                : "")
            );
          }
        },
        tooltipStyle(c)
      ),
      xAxis: Object.assign(
        {
          type: "value",
          max: Math.ceil(maxValue * 1.15),
          axisLabel: baseText(c)
        },
        {
          axisLine: { show: false },
          axisTick: { show: false },
          splitLine: { lineStyle: { color: c.grid } }
        }
      ),
      yAxis: Object.assign(
        {
          type: "category",
          data: rows.map(function (item) {
            return item.topic;
          }),
          axisLabel: Object.assign({}, baseText(c), { fontSize: 12, color: c.inkSoft })
        },
        axisCommon(c)
      ),
      series: [
        {
          type: "bar",
          data: values,
          barWidth: 16,
          itemStyle: { color: c.pine, borderRadius: [0, 3, 3, 0] },
          label: {
            show: true,
            position: "right",
            distance: 8,
            color: c.muted,
            fontSize: 11,
            fontFamily: baseText(c).fontFamily
          }
        }
      ]
    });
  }

  function buildNetwork(c) {
    var chart = initChart("author-network");
    if (!chart) {
      return;
    }
    var network = data.authorNetwork || {};
    var nodes = network.nodes || [];
    var links = network.links || [];
    var categories = network.categories || [];

    chart.setOption({
      animationDuration: reducedMotion ? 0 : 800,
      animationEasing: "cubicOut",
      color: seriesPalette,
      tooltip: Object.assign(
        {
          formatter: function (params) {
            if (params.dataType === "edge") {
              var edge = params.data;
              return (
                "<b>" +
                escapeHtml(edge.source) +
                " ↔ " +
                escapeHtml(edge.target) +
                "</b><br>关联强度 " +
                escapeHtml(edge.value) +
                (edge.relation ? "<br>" + escapeHtml(edge.relation) : "")
              );
            }
            var node = params.data;
            return (
              "<b>" +
              escapeHtml(node.name) +
              "</b><br>回答 " +
              escapeHtml(node.answers) +
              " · 赞同 " +
              escapeHtml(node.upvotes) +
              "<br>主题 " +
              escapeHtml(node.topic) +
              "<br>关键词 " +
              escapeHtml(node.keywords) +
              "<br>平均情感 " +
              escapeHtml(node.avg_sentiment)
            );
          }
        },
        tooltipStyle(c)
      ),
      legend:
        categories.length > 1
          ? [
              {
                top: 0,
                left: "center",
                itemWidth: 9,
                itemHeight: 9,
                textStyle: baseText(c),
                data: categories.map(function (item) {
                  return item.name;
                })
              }
            ]
          : [],
      series: [
        {
          type: "graph",
          layout: "force",
          roam: true,
          draggable: true,
          data: nodes,
          links: links,
          categories: categories,
          top: categories.length > 1 ? 34 : 10,
          bottom: 10,
          label: {
            show: true,
            position: "right",
            color: c.inkSoft,
            fontSize: 10,
            fontFamily: baseText(c).fontFamily,
            formatter: function (params) {
              var name = params.name || "";
              return name.length > 14 ? name.slice(0, 14) + "…" : name;
            }
          },
          labelLayout: { hideOverlap: true, moveOverlap: "shiftY" },
          lineStyle: { color: "source", curveness: 0.14, opacity: 0.45 },
          emphasis: {
            focus: "adjacency",
            lineStyle: { width: 4, opacity: 0.85 }
          },
          force: { repulsion: 430, gravity: 0.06, edgeLength: [100, 230] }
        }
      ]
    });
  }

  function buildAll() {
    if (!window.echarts) {
      emptyNotice();
      return;
    }
    var c = palette();
    charts = [];
    buildKeywords(c);
    buildSentiment(c);
    buildTimeline(c);
    buildScatter(c);
    buildTopics(c);
    buildNetwork(c);
  }

  function resizeAll() {
    charts.forEach(function (chart) {
      if (chart && !chart.isDisposed()) {
        chart.resize();
      }
    });
    syncFrameHeight();
  }

  function syncFrameHeight() {
    var height = document.documentElement.scrollHeight + 28;
    var frame = window.frameElement;
    if (frame) {
      try {
        frame.style.height = height + "px";
      } catch (error) {
        /* Opened directly rather than embedded. */
      }
    }
    // postMessage works even when the parent is a cross-origin preview host.
    if (window.parent && window.parent !== window) {
      try {
        window.parent.postMessage({ type: "zhihu-dashboard-height", height: height }, "*");
      } catch (error) {
        /* Ignore blocked postMessage. */
      }
    }
  }

  applyTheme(readStoredTheme() || systemTheme(), false);

  if (themeToggle) {
    themeToggle.addEventListener("click", function () {
      var next = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
      applyTheme(next, true);
      charts.forEach(function (chart) {
        if (chart && !chart.isDisposed()) {
          chart.dispose();
        }
      });
      buildAll();
    });
  }

  if (media && media.addEventListener) {
    media.addEventListener("change", function () {
      if (!readStoredTheme()) {
        applyTheme(systemTheme(), false);
        charts.forEach(function (chart) {
          if (chart && !chart.isDisposed()) {
            chart.dispose();
          }
        });
        buildAll();
      }
    });
  }

  window.addEventListener("load", function () {
    buildAll();
    syncFrameHeight();
    setTimeout(syncFrameHeight, 350);
    setTimeout(syncFrameHeight, 1100);
  });

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(resizeAll, 120);
  });

  if ("ResizeObserver" in window) {
    var observer = new ResizeObserver(function () {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(syncFrameHeight, 120);
    });
    observer.observe(document.body);
  }
})();
