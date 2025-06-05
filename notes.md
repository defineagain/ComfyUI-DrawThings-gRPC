node...
    - onAdded
    - loadedGraphNode
    - onConnectionChanged (output)
    - server/port changed
    - if workflow fails?
        -> updateNodeModels

updateNodeModels(node)
    - find the sampler node that has server/port
    - fetch the models list and update store
    - update root node models
    - update input nodes
    - if an error option is selected
        - selected lastSelectedModel or (None selected)

    - if models can't be fetched
        - saveSelectedModel for each node
        - set models to:
            - (Couldn't connect to server) <- keep selected
            - (Check server and click to retry)
            - ([lastSelectedModel], disabled)

    - if selected model is not in the list
        - select (None selected)

    - if lora/cnet/upsca/refiner is not connected to sampler
        - set models to
            - ([lastSelectedModel], disabled) <- keep selected
            - (Not connected to sampler node)
            - (Connect to a sampler mode to list available models)

hmmmmm.... when loading the app, updateNodeModels is called many times. 13 times for a sampler + lora + cnet
I already set it up so the models request only happens once (per server:port), maybe I should roll the updates
together too.... premature optimization? But it's an extension, so play nice? shrug I dunno.

---

Okay since I'm in charge of the project now, I wanna do some restructuring of the JS code

There are different kinds of functionality that need to be added, and the js code should be broken up by feature rather than node

dynamic nodes
- dynamic widgets
- dynamic inputs
- right click menu options

model related
- model loading
- dt_model type
- node connections

it's actually mostly broken up like that already.

I'd also like to break up nodes.py, and change the sampler() signature to something more maintainable. It's bad enough that the node method *has* to have 30 params, that have to be kept in the same order as the node definition, but then lining that up with *another* 30-param call and signature is just annoying. I'd rather use a dictionary, and have a default values dict that can be merged.

I'm also not happy with the way dynamic widgets are set up, but I think it's simple enough that I'll leave it as is. Although rather than using the extension hook, I'd rather update the node proto itself, just so it's the same pattern as the model nodes.

It might be weird wrapping a proto method several times from different places, but that seems to be the way comfy is set up:
    - copy the original function
        `const original = node.prototype.onAdded`
    - assign a new function
        `node.prototype.onAdded = function() {`
    - call the original function from the new one
        `const r = original.apply(this, arguments);`
    - do your business
    - return the result from the original call
        `return r`

(confirmed that that is the correct idiom, and also that the best way to extend the nodes is beforeRegisterNode and updating the prototype)

- Nodes
    - Sampler
        - onAdded
            - set up input colors
        - onNodeCreated (?)
            - callbacks for server and port widgets
        - onConfigure
            - updateNodeModels
        - getExtraMenuOptions
        - new members: getServer(), getModelVersion()
        -
    - Prompt
        - onAdded
            - set up colors
    - Lora
        - onAdded, onWidgetChanged (for showing image input on loras that use it)
    - [nodes that use dt_model: Sampler, ControlNet, Lora, Refiner, Upscaler, eventually Prompt]
        - loadedGraphNode
            - check selected option values for "[Object object]"
            - save selected models <- I don't think I need this anymore, since it should be serialized
        - refreshComboInNodes
            - update node models
        - onSerialize
        - onConfigure (deserialize)
        - onConnectionsChange (not for sampler node)
        - new members: saveSelectedModels, lastSelectedModel, isDtServerNode,

- other
    - DT_MODEL type handler

todo
    - remove any extension hooks that operate on nodes, and use beforeRegisterNodeDef instead
        -exceptions:
            - getCustomWidgets
            - loadedGraphNode ?
            - refreshComboInNodes
    - when a new node is connected to a widget input, rename it to match the widget name (so that if/when the widget is hidden, you can still see what the node is controlling) (add an option to disable)
    - (mostly done) improve the DT_MODEL type so that...
        - if disconnected, an error message is shown
        - the previously selected model is saved
        - when reconnected, select the previously selected model
    - replace the two DT prompt nodes with a single node
        - changes color and title depending on the input it's connected to
        - has a button to insert textual inversions into the prompt
        - consider putting multiple text fields with corresponding outputs in the same node (so both prompts can be in the same node)
    - DT Seed node
    - error checking and warnings
        - highlight invalid values (out of range, mismatched model versions, etc)
    - cancellation
    - add options
        - stochastic sampling
        - zero negative prompt
        - (negative) original image size for sdxl

