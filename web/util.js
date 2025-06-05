/**
 * @template {Object} T
 * @template {keyof T} K
 * @param {T} target
 * @param {K} callbackName
 * @param {T[K]} callback
 */

export function setCallback(target, callbackName, callback) {
    const originalCallback = target[callbackName]
    target[callbackName] = function (...args) {
        const r = originalCallback?.apply(this, args)
        callback?.apply(this, args)
        return r
    }
}


/**
 * Updates a prototype with new properties, preserving existing ones. Intended for use with
 * beforeRegisterNodeDef(). See dtModelNodes.js for an example
 *
 * Accepts functions and property descriptors. Functions already on the prototype will be wrapped
 *
 * @param {{prototype: any}} base an object with a prototype to be updated (not the prototype itself)
 * @param {Record<string, Function | PropertyDescriptor | any} update
 */
export function updateProto(base, update) {
    const proto = base.prototype;
    for (const key in update) {
        if (typeof update[key] === "function" && proto[key] !== undefined) {
            const original = proto[key];
            proto[key] = function () {
                const r = original.apply(this, arguments);
                try {
                    update[key].apply(this, arguments);
                }
                finally {
                    return r
                }
            };
        } else if (typeof update[key] === "object") {
            Object.defineProperty(proto, key, update[key]);
        } else proto[key] = update[key];
    }
}
