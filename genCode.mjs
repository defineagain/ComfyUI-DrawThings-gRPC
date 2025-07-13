import cp from "child_process"
import fse from "fs-extra"
import { join } from 'path'

async function updateConfig() {
    console.log('Fetching latest config.fbs')
    const res = await fetch(
        "https://api.github.com/repos/drawthingsai/draw-things-community/contents/Libraries/DataModels/Sources/config.fbs"
    )
    const decoded = Buffer.from((await res.json()).content, "base64").toString(
        "utf-8"
    )

    // remove user defined attributes (indexed) and (primary)
    console.log("removing custom attributes")
    const fbs = decoded
        .replace(/ \(indexed\)/g, "")
        .replace(/ \(primary\)/g, "")

    await fse.ensureDir("resources")
    await fse.writeFile("resources/config.fbs", fbs)

    await useFlatc(async (exec) => {
        console.log("running flatc")
        exec([
            "-I ./resources",
            "--python --gen-object-api --grpc --python-typing --gen-onefile",
            "-o ./src/generated",
            "./resources/config.fbs",
        ].join(" "))
    })
}

async function updateClient() {
    console.log('Fetching latest imageService.proto')
    const res = await fetch(
        "https://api.github.com/repos/drawthingsai/draw-things-community/contents/Libraries/GRPC/Models/Sources/imageService/imageService.proto"
    )
    const decoded = Buffer.from((await res.json()).content, "base64").toString(
        "utf-8"
    )

    await fse.ensureDir("resources")
    await fse.writeFile("resources/imageService.proto", decoded)

    console.log("running protoc")
    cp.execSync(
        [
            "python -m grpc_tools.protoc",
            "--proto_path=./resources",
            "--pyi_out=./src/generated",
            "--python_out=./src/generated",
            "--grpc_python_out=./src/generated",
            "./resources/imageService.proto",
        ].join(" ")
    )
}

if (import.meta.filename === process.argv[1]) {
    updateBoth()
        .then(() => process.exit(0))
        .catch(console.error)
}

async function updateBoth() {
    console.log("removing old generated files")
    fse.emptyDirSync("src/generated")
    await updateConfig()
    await updateClient()
    await fixImports()
    console.log("done")
}

/**
 * @param {string} url
 * @param {(exec: (args) => any) => any} callback
 */
async function useFlatc(callback) {
    let temp
    try {
        temp = fse.mkdtempSync("flatc-")
        console.log("downloading flatc")
        cp.execSync(
            `wget -O ${join(temp, "flatc.zip")} https://github.com/google/flatbuffers/releases/download/v25.2.10/Mac.flatc.binary.zip`,
            { stdio: "ignore" }
        )
        cp.execSync(`unzip -u -d ${temp} ${join(temp, "flatc.zip")}`, { stdio: "ignore" })

        const exec = (args) => {
            return cp.execSync(`${join(temp, "flatc")} ${args}`)
        }
        callback(exec)
    }
    finally {
        if (temp)
            fse.rmSync(temp, { recursive: true })
    }
}

async function fixImports() {
    console.log("fixing imports")
    const files = fse
        .readdirSync("src/generated")
        .filter((f) => f.endsWith(".py") || f.endsWith(".pyi"))

    const names = new Set(files.map((f) => f.replace(".pyi", "").replace(".py", "")))

    for (const file of files) {
        const content = fse.readFileSync(`src/generated/${file}`, "utf-8")
        const fixed = content.replaceAll(
            /from ([\w]+) import ([^ ]+)/g,
            (m, p1, p2) => {
                if (names.has(p1)) {
                    console.log(`changed ${p1} to .${p1} in ${file}`)
                    return `from .${p1} import ${p2}`
                }
                return m
            }
        ).replaceAll(
            /import ([^ ]+) as ([^ ]+)/g,
            (m, p1, p2) => {
                if (names.has(p1)) {
                    console.log(`changed ${p1} to .${p1} in ${file}`)
                    return `from . import ${p1} as ${p2}`
                }
                return m
            }
        )
        fse.writeFileSync(`src/generated/${file}`, fixed)
    }
}
