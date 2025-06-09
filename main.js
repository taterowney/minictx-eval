const table = document.getElementById("results");
//const plusedTable = document.getElementById("plused");
const benchmarkRadio = document.getElementById("Benchmark");
//const chartDom = document.getElementById("chart");
//var chart = echarts.init(chartDom);

const dataUrl = "results.json";

var xhr = new XMLHttpRequest();
xhr.open("GET", "results.json", false); // false makes the request synchronous
xhr.send();

const calcAverage = (a, b) => {
  if (a == null || b == null) {
    return null;
  } else {
    return parseFloat(((parseFloat(a) + parseFloat(b)) / 2).toFixed(1));
  }
};

var data;
/*
After calculating the average, the data should be like this:
data[Model]["pass@1"] = {
  "humaneval": ...,
  "mbpp": ...,
  "humaneval+": ...,
  "mbpp+": ...,
  "average": ...,
  "average+": ...,
}
*/
if (xhr.status === 200) {
  data = JSON.parse(xhr.responseText);
//  Object.keys(data).forEach((key) => {
//    data[key]["pass@1"]["average"] = calcAverage(
//      data[key]["pass@1"]["humaneval"],
//      data[key]["pass@1"]["mbpp"],
//    );
//    data[key]["pass@1"]["average+"] = calcAverage(
//      data[key]["pass@1"]["humaneval+"],
//      data[key]["pass@1"]["mbpp+"],
//    );
//  });
  data = Object.keys(data).map((key) => {
    return {
      Model: key,
      ...data[key],
    };
  });
} else {
  // pop up error window
  alert("Failed to load data from results.json");
}
const globalData = data;

const clearTable = () => {
  table.innerHTML = "";
};


//var chartOption = {
//  legend: {
//    data: ["pass@1*"],
//  },
//  grid: {
//    left: "1%",
//    right: "4%",
//    bottom: "3%",
//    containLabel: true,
//  },
//  xAxis: {
//    name: "Act. Size",
//    type: "category",
//    boundaryGap: false,
//    data: [],
//    axisLabel: {
//      formatter: function (value) {
//        return value + "B";
//      },
//    },
//  },
//  yAxis: {
//    name: "PASS@1 (greedy decoding)",
//    type: "value",
//    show: true,
//    nameTextStyle: {
//      align: "left",
//    },
//    splitLine: {
//      show: true,
//      lineStyle: {
//        type: "dashed",
//      },
//    },
//  },
//  legend: {
//    data: ["base", "instructed"],
//    itemStyle: {
//      opacity: 1.0,
//    },
//  },
//  tooltip: {
//    trigger: "item",
//    axisPointer: {
//      type: "cross",
//    },
//  },
//  series: [
//    {
//      name: "base",
//      type: "scatter",
//      data: [],
//      itemStyle: {
//        color: "#91cc75",
//        opacity: 0.2,
//      },
//      emphasis: {
//        focus: "series",
//      },
//      lineStyle: {
//        width: 2,
//      },
//      markLine: {
//        symbol: "none",
//        emphasis: {
//          label: {
//            position: "middle",
//            formatter: function (params) {
//              return params.data.name;
//            },
//          },
//        },
//        data: [],
//      },
//    },
//    {
//      name: "instructed",
//      type: "scatter",
//      data: [],
//      itemStyle: {
//        color: "#5470c6",
//        opacity: 0.2,
//      },
//      emphasis: {
//        focus: "series",
//      },
//      lineStyle: {
//        width: 2,
//      },
//      markLine: {
//        symbol: "none",
//        emphasis: {
//          label: {
//            position: "middle",
//            formatter: function (params) {
//              return params.data.name;
//            },
//          },
//        },
//        data: [],
//      },
//    },
//  ],
//};

const theaders = ["Model", "Pass Rate"];

// score: 'average', 'humaneval', 'mbpp', 'humaneval+', 'mbpp+'
const displayTable = (table, score) => {
  // filter out Null
  data = globalData
    .filter((row) => {
      return row[score] != null;
    })
    .sort((a, b) => {
      return b[score]["passed"]/b[score]["total"] - a[score]["passed"]/a[score]["total"];
    });
  var thead = document.createElement("thead");
  var headerRow = document.createElement("tr");
  // add rank
  var th = document.createElement("th");
  th.textContent = "#";
  headerRow.appendChild(th);
  // headers
  theaders.forEach(function (header) {
    var th = document.createElement("th");
    th.textContent = header;
    headerRow.appendChild(th);
  });
  thead.appendChild(headerRow);
  table.appendChild(thead);

  var tbody = document.createElement("tbody");
  // add rank
  var rank = 1;
  data.forEach((row) => {
    var dataRow = document.createElement("tr");
    var rankCell = document.createElement("td");
    rankCell.textContent = rank;
    dataRow.appendChild(rankCell);
    var modelCell = document.createElement("td");
    if (rank == 1) {
      modelCell.textContent = "🥇 ";
    } else if (rank == 2) {
      modelCell.textContent = "🥈 ";
    } else if (rank == 3) {
      modelCell.textContent = "🥉 ";
    } else {
      modelCell.textContent = "";
    }
    rank++;
    var modelLink = document.createElement("a");
    modelLink.href = row["link"];
    modelLink.textContent = row["Model"] + " @" + row[score]["n"];
    modelLink.classList.add("link-underline-primary");
    modelLink.classList.add("text-nowrap");
    modelCell.appendChild(modelLink);
    modelCell.classList.add("d-flex");
    modelCell.classList.add("flex-nowrap");
//    var prompted = row["prompted"];
//    var opendata = row["open-data"];
//    if (prompted) {
//      // add a symbol to indicate the model is prompted
//      var promptedSymbol = document.createElement("span");
//      promptedSymbol.textContent = "✨";
//      modelCell.appendChild(promptedSymbol);
//    }
//    if (opendata.toUpperCase() == "FULL") {
//      // add a symbol to indicate the model is fully open-sourced
//      var promptedSymbol = document.createElement("span");
//      promptedSymbol.textContent = "💚";
//      modelCell.appendChild(promptedSymbol);
//    } else if (opendata.toUpperCase() == "PARTIAL") {
//      // add a symbol to indicate the model is partially open-sourced
//      // i.e., a subset of the model implementation is close-sourced
//      var promptedSymbol = document.createElement("span");
//      promptedSymbol.textContent = "💙";
//      modelCell.appendChild(promptedSymbol);
//    }
    dataRow.appendChild(modelCell);
    var passCell = document.createElement("td");
//    passCell.textContent = "⚡";
    passCell.classList.add("text-success");
//    if (table == originTable) {
//      passCell.classList.add("text-danger");
//    } else if (table == plusedTable) {
//      passCell.textContent = "⚡";
//      passCell.classList.add("text-success");
//    }
    passCell.textContent += row[score]["passed"] + "/" + row[score]["total"] + " (" + Math.round((row[score]["passed"] / row[score]["total"]) * 1000) / 10 + "%)";
    dataRow.appendChild(passCell);
    tbody.appendChild(dataRow);
  });
  table.appendChild(tbody);
};

//const displayChart = (score) => {
//  const maxMarkLineModels = 8;
//  // sort first
//  const data = globalData
//    .filter((model) => {
//      return model["pass@1"][score] != null;
//    })
//    .sort((a, b) => {
//      return b["pass@1"][score] - a["pass@1"][score];
//    });
//
//  const sizeList = [
//    ...new Set(
//      data
//        .filter((model) => model["size"] != null)
//        .map((model) => Math.round(model["size"])),
//    ),
//  ].sort((a, b) => {
//    return a - b;
//  });
//
//  chartOption.xAxis.data = sizeList;
//  chartOption.yAxis.max =
//    1 + Math.max(...data.map((model) => model["pass@1"][score]));
//
//  const nonPromptedModels = data.filter(
//    (model) => model["prompted"] == false && model["size"] != null,
//  );
//  const promptedModels = data.filter(
//    (model) => model["prompted"] == true && model["size"] != null,
//  );
//  const nonSizeModels = data.filter(
//    model => model.size === null
//  ).slice(0, maxMarkLineModels);
//
//  nonSizeModels.forEach((model) => {
//    chartOption.series[1].markLine.data.push({
//      name: model["Model"],
//      yAxis: model["pass@1"][score],
//    });
//  });
//
//  [nonPromptedModels, promptedModels].forEach((series, idx) => {
//    series.forEach((model) => {
//      chartOption.series[idx].data.push({
//        name: model["Model"],
//        value: [`${Math.round(model["size"])}`, model["pass@1"][score]],
//      });
//    });
//  });
//
//  const offsets = [[50, 0]]
//    .concat(Array.from({ length: sizeList.length - 2 }, () => [0, 0]))
//    .concat([[-50, 0]]);
//  sizeList.forEach((size, idx) => {
//    const bestNonPromptedModel = nonPromptedModels
//      .filter((model) => Math.round(model["size"]) == size)
//      .sort((a, b) => {
//        return b["pass@1"][score] - a["pass@1"][score];
//      })[0];
//    const bestPromptedModel = promptedModels
//      .filter((model) => Math.round(model["size"]) == size)
//      .sort((a, b) => {
//        return b["pass@1"][score] - a["pass@1"][score];
//      })[0];
//    const hightLightBest = (series, model) => {
//      const point = chartOption.series[series].data.find(
//        (point) => point.name == model["Model"],
//      );
//      point.itemStyle = {
//        opacity: 1.0,
//      };
//      point.label = {
//        show: true,
//        position: "top",
//        offset: offsets[idx],
//        formatter: function (params) {
//          return params.data.name;
//        },
//        color: "inherit",
//      };
//    };
//    if (bestNonPromptedModel) {
//      hightLightBest(0, bestNonPromptedModel);
//    }
//    if (bestPromptedModel) {
//      hightLightBest(1, bestPromptedModel);
//    }
//  });
//
//  chart.setOption(chartOption);
//};

const contextRadio = document.getElementById("context");
const fullRepoRadio = document.getElementById("fullRepo");

contextRadio.addEventListener("click", function () {
  clearTable();
//  displayTable(originTable, "humaneval");
//  displayTable(plusedTable, "humaneval+");
  console.log(table);

  displayTable(table, "premise-selection");
//  clearChart();
//  displayChart("humaneval+");
});

fullRepoRadio.addEventListener("click", function () {
    clearTable();
//    displayTable(originTable, "mbpp");
//    displayTable(plusedTable, "mbpp+");
  displayTable(table, "full");

//    clearChart();
//    displayChart("mbpp+");
    });

displayTable(table, "premise-selection");

window.addEventListener("resize", () => {
  this.chart.resize();
});