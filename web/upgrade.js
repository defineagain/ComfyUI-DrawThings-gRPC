import * as App from "../../scripts/app.js"

/** @type {import("@comfyorg/comfyui-frontend-types").ComfyApp} */
const app = App.app

const dataPath = "./drawthings-grpc/data.json"

export async function checkVersion(currentVersion) {
    const lastVersion = await getLastUsedVersion()
    if (lastVersion === null || lastVersion !== currentVersion) {
        await setLastUsedVersion(currentVersion)

        for (const announcement of announcements) {
            app.extensionManager.toast.add({
                summary: announcement.title,
                detail: announcement.detail,
                severity: 'success',
                life: 0
            })
        }

    }
}

async function getLastUsedVersion() {
    const data = await getUserData()
    return data.lastUsedVersion
}

async function setLastUsedVersion(version) {
    const data = await getUserData()
    data.lastUsedVersion = version
    await app.api.storeUserData(dataPath, data)
}

async function getUserData() {
    try {
        const response = await app.api.getUserData(dataPath)
        if (response.status === 200) {
            const data = await response.json()
            return data
        }
    }
    catch (error) {
        console.error(`Error getting user data: ${error}`)
    }

    const data = {
        lastUsedVersion: null,
    }
    await app.api.storeUserData(dataPath, data)
    return data
}

const announcements = [
    {
        version: "1.6.0",
        title: "DrawThings-gRPC 1.6.0",
        detail: [`The Draw Things Sampler has a new input and corresponding node: Hints!`,
            `Hints are Draw Thing's control images, used by ControlNets, Flux Kontext,`,
            `Hi-Dream E1 and a handful of LoRAs. Use this node to add references images,`,
            `like the "Moodboard" in the Draw Things App.`,
            `\n\nFor ControlNets, images can be added with either the ControlNet node,`,
            `or with the Hints node.  For LoRAs (like Flux Depth), use the Hints node with the`,
            `appropriate hint type. For Flux Kontext or Hi-Dream E1, use the Hints node with`,
            `"Shuffle (Moodboard)" as the hint type.`,
            `\n\nNote: Currently pose or scribble images are not working correctly, but depth or`,
            `moodboard images should work as expected.`].join(' ')
    }
]
