const fs = require("fs");
const path = require("path");

function updatePythonVersion() {
  const packageJson = JSON.parse(fs.readFileSync("package.json", "utf8"));
  const version = packageJson.version;
  const versionLine = `__version__ = "${version}"\n`;
  const initFilePath = path.join("talkbot", "__init__.py");

  const lines = fs.readFileSync(initFilePath, "utf8").split("\n");

  const updatedLines = lines.map((line) =>
    line.startsWith("__version__") ? versionLine : line
  );

  fs.writeFileSync(initFilePath, updatedLines.join("\n"), "utf8");
}

if (require.main === module) {
  updatePythonVersion();
}

module.exports = {
  prepare: (pluginConfig, context) => {
    updatePythonVersion();
  },
};
