const table = document.getElementById("results");
const benchmarkRadio = document.getElementById("Benchmark");


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

*/
if (xhr.status === 200) {
  data = JSON.parse(xhr.responseText);

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


const theaders = ["Model", "Pass Rate"];
const displayTable = (table, score, n) => {
  // filter out Null
  data = globalData
    .filter((row) => {
      return row[score] != null && row[score]["passed"].length > n && row[score]["total"] > 0;
    })
    .sort((a, b) => {
      return b[score]["passed"][n]/b[score]["total"] - a[score]["passed"][n]/a[score]["total"];
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
    // modelLink.textContent = row["Model"] + " @" + row[score]["n"];
    modelLink.textContent = row["Model"]
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
    passCell.textContent += row[score]["passed"][n] + "/" + row[score]["total"] + " (" + Math.round((row[score]["passed"][n] / row[score]["total"]) * 1000) / 10 + "%)";  
    dataRow.appendChild(passCell);
    tbody.appendChild(dataRow);
  });
  table.appendChild(tbody);
};


const contextRadio = document.getElementById("context");
const fullRepoRadio = document.getElementById("fullRepo");

const slider      = document.getElementById("bestOfRange");
const bestOfValue = document.getElementById("bestOfValue");

// compute the largest N across both modes
const maxN = 8;

slider.max   = maxN;
slider.value = 1;
bestOfValue.textContent = "1";

function renderForBestOfN(n) {
  clearTable();
  const key = contextRadio.checked ? "in-file" : "full";
  displayTable(table, key, n-1);
}

slider.addEventListener("input", e => {
  bestOfValue.textContent = e.target.value;
  renderForBestOfN(+e.target.value);
});


contextRadio.addEventListener("click", function () {
  clearTable();
  displayTable(table, "in-file", slider.value - 1);
});

fullRepoRadio.addEventListener("click", function () {
  clearTable();
  displayTable(table, "full", slider.value - 1);
});

displayTable(table, "in-file", slider.value - 1);
