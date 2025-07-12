import cp from "child_process";
import fse from "fs-extra";

async function updateConfig() {
    const res = await fetch(
        "https://api.github.com/repos/drawthingsai/draw-things-community/contents/Libraries/DataModels/Sources/config.fbs"
    );
    const decoded = Buffer.from((await res.json()).content, "base64").toString(
        "utf-8"
    );

    // remove user defined attributes (indexed) and (primary)
    const fbs = decoded
        .replace(/ \(indexed\)/g, "")
        .replace(/ \(primary\)/g, "");

    await fse.ensureDir("resources");
    await fse.writeFile("resources/config.fbs", fbs);

    fse.ensureDirSync("gentemp");
    await getFlatc();
    cp.execSync(
        [
            "./gentemp/flatc",
            "-I ./resources",
            "--python --gen-object-api --grpc --python-typing --gen-onefile",
            "-o ./src/generated",
            "./resources/config.fbs",
        ].join(" ")
    );
    fse.rmSync("gentemp", { recursive: true });
}

async function updateClient() {
    const res = await fetch(
        "https://api.github.com/repos/drawthingsai/draw-things-community/contents/Libraries/GRPC/Models/Sources/imageService/imageService.proto"
    );
    const decoded = Buffer.from((await res.json()).content, "base64").toString(
        "utf-8"
    );

    await fse.ensureDir("resources");
    await fse.writeFile("resources/imageService.proto", decoded);

    cp.execSync(
        [
            "python -m grpc_tools.protoc",
            "--proto_path=./resources",
            "--pyi_out=./src/generated",
            "--python_out=./src/generated",
            "--grpc_python_out=./src/generated",
            "./resources/imageService.proto",
        ].join(" ")
    );
}

if (import.meta.filename === process.argv[1]) {
    updateBoth()
        .then(() => process.exit(0))
        .catch(console.error);
}

async function updateBoth() {
    fse.emptyDirSync("src/generated");
    await updateConfig();
    await updateClient();
    await fixImports();
}

async function getFlatc() {
    cp.execSync(
        "wget -O ./gentemp/flatc.zip https://github.com/google/flatbuffers/releases/download/v25.2.10/Mac.flatc.binary.zip"
    );
    cp.execSync("unzip -u -d ./gentemp ./gentemp/flatc.zip");
}

async function fixImports() {
    const files = fse
        .readdirSync("src/generated")
        .filter((f) => f.endsWith(".py") || f.endsWith(".pyi"));

    const names = new Set(files.map((f) => f.replace(".pyi", "").replace(".py", "")));

    for (const file of files) {
        const content = fse.readFileSync(`src/generated/${file}`, "utf-8");
        const fixed = content.replaceAll(
            /from ([\w]+) import ([^ ]+)/g,
            (m, p1, p2) => {
                if (names.has(p1)) {
                    console.log(`changed ${p1} to .${p1} in ${file}`);
                    return `from .${p1} import ${p2}`;
                }
                return m;
            }
        ).replaceAll(
            /import ([^ ]+) as ([^ ]+)/g,
            (m, p1, p2) => {
                if (names.has(p1)) {
                    console.log(`changed ${p1} to .${p1} in ${file}`);
                    return `from . import ${p1} as ${p2}`;
                }
                return m;
            }
        );
        fse.writeFileSync(`src/generated/${file}`, fixed);
    }
}
