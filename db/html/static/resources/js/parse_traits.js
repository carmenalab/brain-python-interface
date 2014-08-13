function Parameters() {
    this.obj = document.createElement("table");
    this.traits = {};
}
Parameters.prototype.update = function(desc) {
    //Update the parameters descriptor to include the updated values
    for (var name in desc) {
        if (typeof(this.traits[name]) != "undefined" &&
            typeof(desc[name].value) == "undefined") {
            if (this.traits[name].inputs.length > 1) {
                var any = false;
                var tuple = [];
                for (var i = 0; i < this.traits[name].inputs.length; i++) {
                    tuple.push(this.traits[name].inputs[i].value);
                    if (this.traits[name].inputs[i].value)
                        any = true;
                }
                if (any)
                    desc[name].value = tuple;
            } else {
                desc[name].value = this.traits[name].inputs[0].value;
            }
        }
    }
    //clear out the parameters box
    this.obj.innerHTML = "";
    //reinitialize with the updated values
    this.traits = {};
    var func;
    var funcs = {
        "Float":this.add_float,
        "Int": this.add_int,
        "Tuple": this.add_tuple,
        "Array": this.add_array,
        "Instance": this.add_instance,
        "String":this.add_string,
        "Enum":this.add_enum,
        "Bool":this.add_bool,
    }


    this.hidden_inputs = [];
    this.hidden_trait_labels = {};

    for (var name in desc) {
        if (funcs[desc[name]['type']])
            // Bind the function to the current object and call
            funcs[desc[name]['type']].bind(this)(name, desc[name]);
        else
            console.log(desc[name]['type']);
    }

    // console.log(this.hidden_inputs);
    this.hide_attrs();
    // console.log(this.hidden_trait_labels);

}
Parameters.prototype.show_all_attrs = function() {
    for (var attr_name in this.hidden_trait_labels) {
        var label = this.hidden_trait_labels[attr_name];
        label.style.visibility = 'visible';
    }

    for (var k in this.hidden_inputs) {
        var input = this.hidden_inputs[k];
        input.style.visibility = 'visible';
    }
}

Parameters.prototype.hide_attrs = function() {
    for (var attr_name in this.hidden_trait_labels) {
        var label = this.hidden_trait_labels[attr_name];
        label.style.visibility = 'hidden';
    }

    for (var k in this.hidden_inputs) {
        var input = this.hidden_inputs[k];
        input.style.visibility = 'hidden';
    }
}

Parameters.prototype._add = function(name, desc) {
    var trait = document.createElement("tr");
    trait.title = desc;
    var td = document.createElement("td");
    td.className = "param_label";
    trait.appendChild(td);
    var label = document.createElement("label");
    label.innerHTML = name;
    label.setAttribute("for", "param_"+name);
    td.appendChild(label);

    return trait;
}

/*
Function to add an attribute row and label where the 'visibility' attribute of the label can be toggled
*/
Parameters.prototype._add2 = function(name, desc, hidden) {
    var trait = document.createElement("tr");
    trait.title = desc;
    var td = document.createElement("td");
    td.className = "param_label";
    trait.appendChild(td);
    var label = document.createElement("label");
    label.innerHTML = name;
    label.setAttribute("for", "param_"+name);
    
    td.appendChild(label);

    // label.style.visibility = hidden;
    if (hidden === 'hidden') {
        this.hidden_trait_labels[name] = label;
    }

    return trait;
}
Parameters.prototype.add_tuple = function(name, info) {
    var len = info['default'].length;
    var trait = this._add2(name, info['desc'], info['hidden']);
    var wrapper = document.createElement("td");
    wrapper.style.webkitColumnCount = len < 4? len : 4;
    wrapper.style.mozColumnCount = len < 4? len : 4;
    wrapper.style.columnCount = len < 4? len : 4;

    wrapper.style.visibility = info['hidden'];
    trait.appendChild(wrapper);
    this.obj.appendChild(trait);

    this.traits[name] = {"obj":trait, "inputs":[]};
    for (var i=0; i < len; i++) {
        var input = document.createElement("input");
        input.type = "text";
        input.name = name;
        input.placeholder = JSON.stringify(info['default'][i]);
        if (typeof(info['value']) != "undefined")
            if (typeof(info['value'][i]) != "string")
                input.value = JSON.stringify(info['value'][i]);
            else
                input.value = info['value'][i];
        // input.style.visibility = info['hidden'];
        if (info['hidden'] === 'hidden') {
            this.hidden_inputs.push(input);
        }

        wrapper.appendChild(input);
        this.traits[name]['inputs'].push(input);
    }
    this.traits[name].inputs[0].id = "param_"+name;
    for (var i in this.traits[name].inputs) {
        var inputs = this.traits[name].inputs
        this.traits[name].inputs[i].onchange = function() {
            if (this.value.length > 0) {
                for (var j in inputs)
                    inputs[j].required = "required";
            } else {
                for (var j in inputs)
                    inputs[j].removeAttribute("required");
            }
        }
    }
}

Parameters.prototype.add_int = function (name, info) {
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "number";
    input.name = name;
    input.id = "param_"+name;
    input.title = "An integer value"
    if (typeof(info['value']) != "undefined")
        input.value = info['value'];
    else
        input.value = info['default'];
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_float = function (name, info) {
    //console.log(info)
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }

    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.title = "A floating point value";
    input.pattern = "-?[0-9]*\.?[0-9]*";
    input.placeholder = info['default'];
    if (typeof(info['value']) == "string")
        input.value = info.value;
    else if (typeof(info['value']) != "undefined")
        input.value = JSON.stringify(info.value);
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_bool = function (name, info) {
    //console.log(info)
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "checkbox";
    input.name = name;
    input.id = "param_"+name;
    input.title = "A boolean value"
    if (typeof(info['value']) != "undefined")
        input.checked=info['value'];
    else
        input.checked = info['default'];
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_array = function (name, info) {
    if (info['default'].length < 4) {
        this.add_tuple(name, info);
        for (var i=0; i < this.traits[name].inputs.length; i++)
            this.traits[name].inputs[i].pattern = '[0-9\\(\\)\\[\\]\\.\\,\\s\\-]*';
    } else {
        var trait = this._add2(name, info['desc'], info['hidden']);
        var div = document.createElement("td");
        var input = document.createElement("input");
        // input.style.visibility = info['hidden'];
        if (info['hidden'] === 'hidden') {
            this.hidden_inputs.push(input);
        }
        trait.appendChild(div);
        div.appendChild(input);
        this.obj.appendChild(trait);

        input.type = "text";
        input.name = name;
        input.id = "param_"+name;
        input.title = "An array value";
        input.placeholder = info['default'];
        if (typeof(info['value']) == "string")
            input.value = info['value'];
        else if (typeof(info['value']) != "undefined")
            input.value = JSON.stringify(info['value']);
        input.pattern = /[0-9\(\)\[\]\.\,\s\-]*/;
        this.traits[name] = {"obj":trait, "inputs":[input]};
    }
}
Parameters.prototype.add_string = function (name, info) {
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("input");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.type = "text";
    input.name = name;
    input.id = "param_"+name;
    input.placeholder = info['default'];
    if (typeof(info['value']) != "undefined") {
        input.setAttribute("value", info['value']);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_instance = function(name, info) {
    //console.log(info)
    var options = info['options'];
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("select");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i][0];
        opt.innerHTML = options[i][1];
        if ((typeof(info['value']) != "undefined" && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.add_enum = function(name, info) {
    //console.log(info)
    var options = info['options'];
    var trait = this._add2(name, info['desc'], info['hidden']);
    var div = document.createElement("td");
    var input = document.createElement("select");
    // input.style.visibility = info['hidden'];
    if (info['hidden'] === 'hidden') {
        this.hidden_inputs.push(input);
    }
    trait.appendChild(div);
    div.appendChild(input);
    this.obj.appendChild(trait);

    input.name = name;
    input.id = "param_"+name;
    for (var i = 0; i < options.length; i++) {
        var opt = document.createElement("option");
        opt.value = options[i];
        opt.innerHTML = options[i];
        if ((typeof(info['value']) != "undefined" && info['value'] == opt.value) ||
            (info['default'] == opt.value))
            opt.setAttribute("selected", "selected");
        input.appendChild(opt);
    }
    this.traits[name] = {"obj":trait, "inputs":[input]};
}
Parameters.prototype.to_json = function() {
    var jsdata = {};
    if (typeof(this.traits["arm_visible"]) != "undefined")
        this.traits["arm_visible"].inputs[0].value = this.traits["arm_visible"].inputs[0].checked;
    for (var name in this.traits) {
        if (this.traits[name].inputs.length > 1) {
            var plist = [];
            for (var i = 0; i < this.traits[name].inputs.length; i++) {
                plist.push(this.traits[name].inputs[i].value)
            }
            if (plist[0].length > 0)
                jsdata[name] = plist;
        } else if (this.traits[name].inputs[0].value.length > 0) {
            jsdata[name] = this.traits[name].inputs[0].value;
        }
    }
    return jsdata;
}