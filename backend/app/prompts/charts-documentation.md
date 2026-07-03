# Chart Generation Reference

## Supported chart types (pyecharts)

### Basic Charts

bar, line, pie, scatter, effectScatter, radar, gauge, funnel, candlestick, heatmap, wordcloud, graph, treemap, sunburst, boxplot, parallel, sankey, chord, liquidfill, tree, calendar, polar, pictorialBar, themeRiver, custom

### Composite Charts

grid (multiple charts in one canvas), page (multi-page), tab (tabbed), timeline (animated timeline)

### 3D Charts

bar3D, line3D, scatter3D, surface3D, lines3D

## Forbidden chart types

**geo, map, amap, bmap, gmap, lmap, map3D, mapGlobe** — these require external map tile downloads and will NOT work in offline mode.

## Format rules

- Always include `title` and `tooltip` fields
- `tooltip.trigger`: use `"axis"` for bar/line/scatter/effectScatter/heatmap/boxplot/candlestick/parallel/calendar, use `"item"` for pie/gauge/funnel/sankey/chord/graph/sunburst/treemap/tree, use `"axis"` for 3D charts
- For categorical data, use `xAxis.type: "category"` with `xAxis.data` array
- For numeric axes, use `yAxis.type: "value"`
- `series` is an array — each object is one data series
- Keep JSON valid — no trailing commas, no comments
- The entire chart must be output as a single JSON object inside an `echarts` code block

## Chart examples

### Bar Chart

```echarts
{
  "title": {"text": "Monthly Revenue", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "category", "data": ["Jan", "Feb", "Mar", "Apr", "May"]},
  "yAxis": {"type": "value"},
  "series": [{"name": "Revenue", "type": "bar", "data": [120, 200, 150, 80, 170]}]
}
```

### Line Chart (single series)

```echarts
{
  "title": {"text": "Temperature Trend", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "category", "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
  "yAxis": {"type": "value"},
  "series": [{"name": "Temp", "type": "line", "data": [22, 25, 23, 28, 30, 27, 24], "smooth": true}]
}
```

### Line Chart (multiple series)

```echarts
{
  "title": {"text": "Sales vs Profit", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "category", "data": ["Q1", "Q2", "Q3", "Q4"]},
  "yAxis": {"type": "value"},
  "series": [
    {"name": "Sales", "type": "line", "data": [120, 200, 150, 80], "smooth": true},
    {"name": "Profit", "type": "line", "data": [40, 80, 60, 30], "smooth": true}
  ]
}
```

### Bar + Line Combo

```echarts
{
  "title": {"text": "Revenue and Growth", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "category", "data": ["Jan", "Feb", "Mar", "Apr"]},
  "yAxis": [{"type": "value", "name": "Revenue"}, {"type": "value", "name": "Growth"}],
  "series": [
    {"name": "Revenue", "type": "bar", "data": [120, 200, 150, 80]},
    {"name": "Growth", "type": "line", "data": [10, 25, 15, 5]}
  ]
}
```

### Pie Chart

```echarts
{
  "title": {"text": "Market Share", "left": "center"},
  "tooltip": {"trigger": "item"},
  "series": [{
    "type": "pie",
    "radius": ["40%", "70%"],
    "data": [
      {"name": "Product A", "value": 100},
      {"name": "Product B", "value": 200},
      {"name": "Product C", "value": 150},
      {"name": "Product D", "value": 80}
    ]
  }]
}
```

### Donut Chart

```echarts
{
  "title": {"text": "Budget Distribution", "left": "center"},
  "tooltip": {"trigger": "item"},
  "series": [{
    "type": "pie",
    "radius": ["50%", "75%"],
    "data": [
      {"name": "Engineering", "value": 40},
      {"name": "Marketing", "value": 25},
      {"name": "Operations", "value": 20},
      {"name": "R&D", "value": 15}
    ]
  }]
}
```

### Scatter Plot

```echarts
{
  "title": {"text": "Revenue vs Ad Spend", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "value"},
  "yAxis": {"type": "value"},
  "series": [{
    "name": "Campaigns",
    "type": "scatter",
    "data": [[10, 20], [20, 40], [30, 55], [40, 70], [50, 90]]
  }]
}
```

### Scatter with Size Encoding

```echarts
{
  "title": {"text": "Population by GDP", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "value", "name": "GDP (B)"},
  "yAxis": {"type": "value", "name": "Population (M)"},
  "series": [{
    "name": "Countries",
    "type": "scatter",
    "data": [
      [5.9, 330], [2.1, 140], [1.7, 130], [1.3, 80], [0.8, 50]
    ],
    "symbolSize": function(data) { return data[2] || 10; }
  }]
}
```

### Effect Scatter (with ripple animation)

```echarts
{
  "title": {"text": "Network Activity", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "value"},
  "yAxis": {"type": "value"},
  "series": [{
    "name": "Events",
    "type": "effectScatter",
    "data": [[10, 20], [30, 50], [50, 30], [70, 80], [90, 60]]
  }]
}
```

### Radar Chart

```echarts
{
  "title": {"text": "Skill Assessment", "left": "center"},
  "tooltip": {},
  "radar": {
    "indicator": [
      {"name": "Python", "max": 100},
      {"name": "JavaScript", "max": 100},
      {"name": "DevOps", "max": 100},
      {"name": "ML", "max": 100},
      {"name": "DB", "max": 100},
      {"name": "API", "max": 100}
    ]
  },
  "series": [{
    "type": "radar",
    "data": [{
      "value": [90, 70, 80, 85, 75, 90],
      "name": "Current"
    }]
  }]
}
```

### Gauge Chart

```echarts
{
  "title": {"text": "Server Load", "left": "center"},
  "series": [{
    "type": "gauge",
    "max": 100,
    "detail": {"formatter": "{value}%"},
    "data": [{"value": 75, "name": "CPU Usage"}]
  }]
}
```

### Gauge with Multiple Metrics

```echarts
{
  "title": {"text": "System Health", "left": "center"},
  "series": [
    {"type": "gauge", "max": 100, "data": [{"value": 75, "name": "CPU"}]},
    {"type": "gauge", "max": 100, "data": [{"value": 60, "name": "Memory"}]},
    {"type": "gauge", "max": 100, "data": [{"value": 45, "name": "Disk"}]}
  ]
}
```

### Funnel Chart

```echarts
{
  "title": {"text": "Sales Pipeline", "left": "center"},
  "tooltip": {"trigger": "item"},
  "series": [{
    "type": "funnel",
    "data": [
      {"name": "Leads", "value": 1000},
      {"name": "Qualified", "value": 600},
      {"name": "Proposal", "value": 300},
      {"name": "Negotiation", "value": 150},
      {"name": "Closed", "value": 80}
    ]
  }]
}
```

### Candlestick (OHLC) Chart

```echarts
{
  "title": {"text": "Stock Price", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "category", "data": ["2024-01", "2024-02", "2024-03", "2024-04"]},
  "yAxis": {"type": "value"},
  "series": [{
    "name": "AAPL",
    "type": "candlestick",
    "data": [[20, 30, 15, 40], [30, 50, 25, 55], [50, 60, 40, 70], [60, 80, 55, 75]]
  }]
}
```

### Heatmap Chart

```echarts
{
  "title": {"text": "Activity Heatmap", "left": "center"},
  "tooltip": {},
  "xAxis": {"type": "category", "data": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]},
  "yAxis": {"type": "category", "data": ["12am", "4am", "8am", "12pm", "4pm", "8pm"]},
  "visualMap": {"min": 0, "max": 10, "calculable": true},
  "series": [{
    "type": "heatmap",
    "data": [[0,0,5],[0,1,3],[0,2,7],[0,3,2],[0,4,8],[0,5,1],[1,0,4],[1,1,6],[1,2,2],[1,3,9],[1,4,3],[1,5,7],[2,0,8],[2,1,1],[2,2,5],[2,3,4],[2,4,2],[2,5,6],[3,0,3],[3,1,7],[3,2,8],[3,3,1],[3,4,9],[3,5,4],[4,0,6],[4,1,2],[4,2,4],[4,3,7],[4,4,5],[4,5,8],[5,0,1],[5,1,9],[5,2,3],[5,3,6],[5,4,7],[5,5,2],[6,0,5],[6,1,4],[6,2,1],[6,3,8],[6,4,6],[6,5,3]],
    "label": {"show": true}
  }]
}
```

### Word Cloud

```echarts
{
  "title": {"text": "Key Topics", "left": "center"},
  "series": [{
    "type": "wordCloud",
    "data": [
      {"name": "Python", "value": 100},
      {"name": "Machine Learning", "value": 80},
      {"name": "Data Science", "value": 70},
      {"name": "AI", "value": 60},
      {"name": "Cloud", "value": 50},
      {"name": "DevOps", "value": 40},
      {"name": "Kubernetes", "value": 30},
      {"name": "Docker", "value": 25}
    ]
  }]
}
```

### Graph (Network)

```echarts
{
  "title": {"text": "Network Topology", "left": "center"},
  "tooltip": {},
  "series": [{
    "type": "graph",
    "layout": "force",
    "data": [
      {"name": "Node A"}, {"name": "Node B"}, {"name": "Node C"},
      {"name": "Node D"}, {"name": "Node E"}
    ],
    "links": [
      {"source": "Node A", "target": "Node B"},
      {"source": "Node B", "target": "Node C"},
      {"source": "Node C", "target": "Node D"},
      {"source": "Node D", "target": "Node E"},
      {"source": "Node E", "target": "Node A"}
    ]
  }]
}
```

### Treemap

```echarts
{
  "title": {"text": "Disk Usage", "left": "center"},
  "series": [{
    "type": "treemap",
    "data": [
      {"name": "System", "value": 50, "children": [
        {"name": "OS", "value": 30},
        {"name": "Apps", "value": 20}
      ]},
      {"name": "Documents", "value": 30, "children": [
        {"name": "Work", "value": 20},
        {"name": "Personal", "value": 10}
      ]},
      {"name": "Media", "value": 20}
    ]
  }]
}
```

### Sunburst

```echarts
{
  "title": {"text": "Revenue Breakdown", "left": "center"},
  "series": [{
    "type": "sunburst",
    "data": [
      {"name": "Products", "value": 100, "children": [
        {"name": "Software", "value": 60, "children": [
          {"name": "SaaS", "value": 40},
          {"name": "License", "value": 20}
        ]},
        {"name": "Hardware", "value": 40}
      ]},
      {"name": "Services", "value": 50, "children": [
        {"name": "Consulting", "value": 30},
        {"name": "Support", "value": 20}
      ]}
    ]
  }]
}
```

### Boxplot

```echarts
{
  "title": {"text": "Test Scores by Class", "left": "center"},
  "tooltip": {"trigger": "item"},
  "xAxis": {"type": "category", "data": ["Class A", "Class B", "Class C"]},
  "yAxis": {"type": "value"},
  "series": [{
    "type": "boxplot",
    "data": [
      [[60, 70, 75, 85, 95], [55, 65, 72, 80, 90], [50, 60, 68, 78, 88]],
      [[40, 55, 60, 70, 80], [45, 58, 63, 72, 82], [35, 50, 57, 67, 77]],
      [[70, 80, 85, 90, 98], [65, 75, 82, 88, 96], [60, 72, 78, 85, 94]]
    ]
  }]
}
```

### Parallel Coordinates

```echarts
{
  "title": {"text": "Employee Profiles", "left": "center"},
  "tooltip": {},
  "parallel": {
    "top": 50,
    "bottom": 50,
    "left": 100,
    "right": 100
  },
  "parallelAxis": [
    {"dim": 0, "name": "Experience"},
    {"dim": 1, "name": "Skills"},
    {"dim": 2, "name": "Education"},
    {"dim": 3, "name": "Performance"}
  ],
  "series": [{
    "type": "parallel",
    "data": [[[3, 70, 60, 80], [5, 85, 80, 90], [2, 60, 50, 65], [7, 90, 90, 95]]]
  }]
}
```

### Sankey Diagram

```echarts
{
  "title": {"text": "Data Flow", "left": "center"},
  "series": [{
    "type": "sankey",
    "data": [
      {"name": "Source A"}, {"name": "Source B"},
      {"name": "Node C"}, {"name": "Node D"},
      {"name": "Target E"}, {"name": "Target F"}
    ],
    "links": [
      {"source": "Source A", "target": "Node C", "value": 50},
      {"source": "Source B", "target": "Node C", "value": 30},
      {"source": "Source A", "target": "Node D", "value": 20},
      {"source": "Node C", "target": "Target E", "value": 40},
      {"source": "Node C", "target": "Target F", "value": 20},
      {"source": "Node D", "target": "Target E", "value": 10},
      {"source": "Node D", "target": "Target F", "value": 30}
    ]
  }]
}
```

### Chord Diagram

```echarts
{
  "title": {"text": "Trade Relations", "left": "center"},
  "series": [{
    "type": "chord",
    "data": [
      {"name": "Region A"}, {"name": "Region B"}, {"name": "Region C"}
    ],
    "links": [
      {"source": "Region A", "target": "Region B", "value": 30},
      {"source": "Region B", "target": "Region C", "value": 20},
      {"source": "Region C", "target": "Region A", "value": 15}
    ]
  }]
}
```

### Liquid Fill Chart

```echarts
{
  "title": {"text": "Server Capacity", "left": "center"},
  "series": [{
    "type": "liquidFill",
    "data": [0.7, 0.6, 0.5],
    "radius": "70%"
  }]
}
```

### Tree Diagram

```echarts
{
  "title": {"text": "Organization Chart", "left": "center"},
  "series": [{
    "type": "tree",
    "data": [{
      "name": "CEO",
      "children": [
        {"name": "CTO", "children": [{"name": "Dev Lead"}, {"name": "QA Lead"}]},
        {"name": "CFO", "children": [{"name": "Accountant"}, {"name": "Auditor"}]}
      ]
    }],
    "layout": "orthogonal",
    "orient": "vertical"
  }]
}
```

### Calendar Heatmap

```echarts
{
  "title": {"text": "2024 Activity", "left": "center"},
  "tooltip": {},
  "calendar": {"top": 50, "bottom": 50, "range": "2024"},
  "visualMap": {"min": 0, "max": 10, "calculable": true, "orient": "horizontal", "left": "center", "bottom": 10},
  "series": [{
    "type": "heatmap",
    "coordinateSystem": "calendar",
    "data": [
      ["2024-01-15", 5], ["2024-01-16", 3], ["2024-01-17", 8],
      ["2024-02-10", 6], ["2024-02-11", 2], ["2024-02-12", 9],
      ["2024-03-05", 4], ["2024-03-06", 7], ["2024-03-07", 1]
    ]
  }]
}
```

### Polar Chart

```echarts
{
  "title": {"text": "Performance Metrics", "left": "center"},
  "tooltip": {},
  "polar": {},
  "angleAxis": {"type": "category", "data": ["A", "B", "C", "D", "E"]},
  "radiusAxis": {"type": "value"},
  "series": [{
    "type": "bar",
    "data": [65, 80, 70, 90, 55],
    "coordinateSystem": "polar",
    "name": "Score"
  }]
}
```

### Pictorial Bar Chart

```echarts
{
  "title": {"text": "Sales Units", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "category", "data": ["Jan", "Feb", "Mar", "Apr"]},
  "yAxis": {"type": "value"},
  "series": [{
    "type": "pictorialBar",
    "data": [120, 200, 150, 80],
    "symbolSize": ["auto", "auto"],
    "symbolMargin": "25%"
  }]
}
```

### ThemeRiver

```echarts
{
  "title": {"text": "Topic Evolution", "left": "center"},
  "tooltip": {"trigger": "axis"},
  "xAxis": {"type": "time"},
  "yAxis": {},
  "visualMap": {"top": 10, "right": 10, "min": 0, "max": 100, "inRange": {"color": ["#313695", "#4575b4", "#74add1", "#abd9e9", "#fdae61", "#f46d43", "#d73027"]}},
  "series": [{
    "type": "themeRiver",
    "data": [
      ["2024-01-01", 30, "Topic A"],
      ["2024-01-02", 40, "Topic A"],
      ["2024-01-01", 20, "Topic B"],
      ["2024-01-02", 35, "Topic B"],
      ["2024-01-01", 15, "Topic C"],
      ["2024-01-02", 25, "Topic C"]
    ]
  }]
}
```

### Bar3D (3D Bar Chart)

```echarts
{
  "title": {"text": "3D Sales Data", "left": "center"},
  "tooltip": {},
  "xAxis3D": {"type": "category", "data": ["A", "B", "C", "D"]},
  "yAxis3D": {"type": "category", "data": ["Q1", "Q2", "Q3", "Q4"]},
  "grid3D": {},
  "series": [{
    "type": "bar3D",
    "data": [[0,0,120],[0,1,200],[0,2,150],[0,3,80],[1,0,90],[1,1,160],[1,2,110],[1,3,60]]
  }]
}
```

### Line3D (3D Line Chart)

```echarts
{
  "title": {"text": "3D Trend", "left": "center"},
  "tooltip": {},
  "xAxis3D": {"type": "value"},
  "yAxis3D": {"type": "value"},
  "zAxis3D": {"type": "value"},
  "grid3D": {},
  "series": [{
    "type": "line3D",
    "data": [[10,20,30],[20,40,50],[30,55,70],[40,70,85]]
  }]
}
```

### Scatter3D (3D Scatter Plot)

```echarts
{
  "title": {"text": "3D Points", "left": "center"},
  "tooltip": {},
  "xAxis3D": {"type": "value"},
  "yAxis3D": {"type": "value"},
  "zAxis3D": {"type": "value"},
  "grid3D": {},
  "series": [{
    "type": "scatter3D",
    "data": [[10,20,30],[20,40,50],[30,55,70],[40,70,85],[50,90,100]]
  }]
}
```

### Surface3D (3D Surface)

```echarts
{
  "title": {"text": "3D Surface", "left": "center"},
  "tooltip": {},
  "xAxis3D": {"type": "value"},
  "yAxis3D": {"type": "value"},
  "zAxis3D": {"type": "value"},
  "grid3D": {},
  "series": [{
    "type": "surface",
    "data": [[[1,1,10],[2,1,20],[3,1,30]],[[1,2,15],[2,2,25],[3,2,35]],[[1,3,20],[2,3,30],[3,3,40]]]
  }]
}
```

### Lines3D (3D Lines)

```echarts
{
  "title": {"text": "3D Flight Paths", "left": "center"},
  "tooltip": {},
  "xAxis3D": {"type": "value"},
  "yAxis3D": {"type": "value"},
  "zAxis3D": {"type": "value"},
  "grid3D": {},
  "series": [{
    "type": "lines3D",
    "data": [[[0,0,0],[10,10,10]],[[0,0,0],[20,5,15]]]
  }]
}
```

### Grid (Multiple Charts on One Canvas)

```echarts
{
  "title": {"text": "Dashboard View", "left": "center"},
  "grid": [{"top": "10%", "left": "5%", "width": "40%", "height": "40%"}, {"top": "10%", "right": "5%", "width": "40%", "height": "40%"}],
  "xAxis": [{"type": "category", "data": ["Jan","Feb","Mar"], "gridIndex": 0}, {"type": "category", "data": ["Jan","Feb","Mar"], "gridIndex": 1}],
  "yAxis": [{"type": "value", "gridIndex": 0}, {"type": "value", "gridIndex": 1}],
  "series": [
    {"type": "bar", "xAxisIndex": 0, "yAxisIndex": 0, "data": [120, 200, 150]},
    {"type": "line", "xAxisIndex": 1, "yAxisIndex": 1, "data": [80, 120, 90]}
  ]
}
```

### Timeline (Animated Sequence)

```echarts
{
  "title": {"text": "Revenue Over Time", "left": "center"},
  "tooltip": {},
  "timeline": {"data": ["2020", "2021", "2022", "2023"]},
  "series": [{"type": "bar", "data": [100, 150, 200, 250]}]
}
```

## Output format

When generating a chart, output it as an echarts code block:

```echarts
{chart JSON here}
```

The code block must use the exact fence format: three backticks, the word "echarts", then the JSON, then three backticks on a new line.
