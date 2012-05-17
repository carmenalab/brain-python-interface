function Parameters(desc) {
    this.obj = document.createElement("div");
    this.traits = {};
    var funcs = {
        "Float":this.add_float,
        "Int": this.add_int,
        "Tuple": this.add_tuple,
        "Array": this.add_array,
    }
    for (var name in desc) {
        if (funcs[desc[name]['type']]) {
            var func = funcs[desc[name]['type']].bind(this);
            func(name, desc[name]['desc'], desc[name]['default']);
        } else
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
Parameters.prototype.add_tuple = function(name, desc, value) {
    var trait = this._add(name, desc);
    this.traits[name] = {"obj":trait, "inputs":[]};

    for (var i=0; i < value.length; i++) {
        var input = document.createElement("input");
        input.type = "text";
        input.name = name+"["+i+"]";
        input.pattern = "[\d\.\-]*";
        input.placeholder = value[i];
        input.title = "A floating point value";

        trait.appendChild(input);
        this.traits[name]['inputs'].push(input);
    }
    this.traits[name]['inputs'][0].id = "param_"+name;
    console.log(trait.firstChild);
    this.obj.appendChild(trait);
}
Parameters.prototype.add_int = function (name, desc, value) {
    var trait = this._add(name, desc);
    var input = document.createElement("input");
    input.value = value;
    input.type = "number";
    input.name = name;
    input.id = "param_"+name;
    input.title = "An integer value"
    trait.appendChild(input);
    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}

Parameters.prototype.add_float = function (name, desc, value) {
    var trait = this._add(name, desc);
    var input = document.createElement("input");
    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.title = "A floating point value";
    input.pattern = "[\d\.\-]*";
    input.placeholder = value;
    trait.appendChild(input);

    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}
Parameters.prototype.add_array = function (name, desc, value) {
    if (value.length < 4) 
        return this.add_tuple(name, desc, value);
    var trait = this._add(name, desc);
    var input = document.createElement("input");
    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.title = "An array value";
    input.placeholder = value;
    input.pattern = "[\(\)\[\]\d\.\,\s\-]*";
    trait.appendChild(input);
    this.traits[name] = {"obj":trait, "inputs":[input]};
    this.obj.appendChild(trait);
}