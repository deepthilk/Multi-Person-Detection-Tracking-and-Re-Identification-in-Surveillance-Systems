const inputFile = document.getElementById("inputFile");
const outputFile = document.getElementById("outputFile");
const inputVideo = document.getElementById("inputVideo");
const outputVideo = document.getElementById("outputVideo");
const inputName = document.getElementById("inputName");
const outputName = document.getElementById("outputName");
const inputPlaceholder = document.getElementById("inputPlaceholder");
const outputPlaceholder = document.getElementById("outputPlaceholder");
const swapBtn = document.getElementById("swapBtn");

const setVideo = (file, videoEl, labelEl, placeholderEl) => {
  if (!file) {
    labelEl.textContent = "None";
    return;
  }

  const url = URL.createObjectURL(file);
  videoEl.src = url;
  videoEl.load();
  placeholderEl.style.display = "none";
  labelEl.textContent = file.name;
};

inputFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setVideo(file, inputVideo, inputName, inputPlaceholder);
});

outputFile.addEventListener("change", (event) => {
  const [file] = event.target.files;
  setVideo(file, outputVideo, outputName, outputPlaceholder);
});

swapBtn.addEventListener("click", () => {
  const inputParent = document.querySelector(".screen[data-role='input']");
  const outputParent = document.querySelector(".screen[data-role='output']");
  const stage = document.querySelector(".stage");

  stage.insertBefore(outputParent, inputParent);
});
