function Parameters(desc) {
    this.obj = document.createElement("div");
    this.traits = {};
    var func;
    var funcs = {
        "Float":this.add_float,
        "Int": this.add_int,
        "Tuple": this.add_tuple,
        "Array": this.add_array,
        "Instance": this.add_instance,
    }
    for (var name in desc) {
        if (funcs[desc[name]['type']])
            funcs[desc[name]['type']].bind(this)(name, desc[name]);
        else
            console.log(desc[name]['type']);
    }
}
Parameters.prototype._add = function(name, desc) {
    var trait = document.createElement("div");
    trait.title = desc;

    var label = document.createElement("label");
    label.innerHTML = name;
    label.setAttribute("for", "param_"+name);
    trait.appendChild(label);

    return trait;
}
Parameters.prototype.add_tuple = function(name, info) {
    var trait = this._add(name, info['desc']);
    this.traits[name] = {"obj":trait, "inputs":[]};

    for (var i=0; i < info['default'].length; i++) {
        var input = document.createElement("input");
        input.type = "text";
        input.name = name+"["+i+"]";
        input.pattern = "[\d\.\-]*";
        input.placeholder = info['default'][i];
        input.title = "A floating point value";
        if (info['value'])
            input.value = input['value'][i];

        trait.appendChild(input);
        this.traits[name]['inputs'].push(input);
    }
    this.traits[name]['inputs'][0].id = "param_"+name;
    console.log(trait.firstChild);
    this.obj.appendChild(trait);
}
Parameters.prototype.add_int = function (name, info) {
    var trait = this._add(name, info['desc']);
    var input = document.createElement("input");
    input.type = "number";
    input.name = name;
    input.id = "param_"+name;
    input.title = "An integer value"
    if (info['value'])
        input.value = input['value'];
    else
        input.value = input['default'];
    trait.appendChild(input);
    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}

Parameters.prototype.add_float = function (name, info) {
    var trait = this._add(name, info['desc']);
    var input = document.createElement("input");
    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.title = "A floating point value";
    input.pattern = "[\d\.\-]*";
    input.placeholder = info['default'];
    if (info['value'])
        input.value = input['value'];
    trait.appendChild(input);

    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}
Parameters.prototype.add_array = function (name, info) {
    if (info['default'].length < 4) 
        return this.add_tuple(name, info);

    var trait = this._add(name, info['desc']);
    var input = document.createElement("input");
    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.title = "An array value";
    input.placeholder = info['default'];
    if (info['value'])
        input.value = input['value'];
    input.pattern = "[\(\)\[\]\d\.\,\s\-]*";
    trait.appendChild(input);
    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}
Parameters.prototype.add_instance = function(name, info) {
    var options = info['options'];
    var trait = this._add(name, info['desc']);
    var input = document.createElement("select");
    input.name = name;
    input.id = "param_"+name;
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i][0];
        opt.innerHTML = options[i][1];
        if ((info['value'] && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    trait.appendChild(input);
    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}
Parameters.prototype.to_json = function() {
    var jsdata = {};
    for (var name in this.traits) {
        if (this.traits[name]['inputs'].length > 1) {
            var plist = [];
            for (var i = 0; i < this.traits[name]['inputs'].length; i++) {
                plist.push(this.traits[name]['inputs'][i].value);
            }
            jsdata[name] = plist;
        } else {
            jsdata[name] = this.traits[name]['inputs'][0].value;
        }
    }
    return jsdata;
}