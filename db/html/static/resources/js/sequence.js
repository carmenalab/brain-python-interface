function Sequence(info) {
    var _this = this;
    this.params = new Parameters({});
    $("#seqparams").append(this.params.obj);
    this.init(info);
    $("#seqgen").change(this._generator.bind(this));
    $("#seqlist").change(function () {
        var id = this.value;
        _this.params.update(info[id].params);
        $("#seqstatic").attr("checked", info[id].static);
    })
}
Sequence.prototype.init = function(info) {
    this.options = {};
    var opt, id;
    for (id in info) {
        opt = document.createElement("option");
        opt.innerHTML = info[id].name;
        opt.value = id;
        this.options[id] = opt;
        $("#seqlist").append(opt);
    }
    $("#seqgen option").filter(":selected").removeAttr("selected");
    if (typeof(id) != "undefined") {
        var genid = info[id].generator[0]
        $("#seqgen option").each(function() {
            if (this.value == genid)
                this.selected = "selected";
        })
        $("#seqlist option").each(function() {
            if (this.value == id)
                this.selected = "selected";
        })
        this.params.update(info[id].params);
        $("#seqstatic").attr("checked", info[id].static);
    } else {
        this._generator();
    }
}
Sequence.prototype.update = function(info) {
    for (var id in this.options)
        $(this.options[id]).remove()
    this.init(info);
}
Sequence.prototype.destroy = function() {
    for (var id in this.options)
        $(this.options[id]).remove()
    $(this.params.obj).remove()
    delete this.params
    $("#seqlist").unbind("change");
    $("#seqgen").unbind("change");
}
Sequence.prototype._generator = function() {
    var genid = $("#seqgen").attr("value");
    $.getJSON("ajax/gen_info/"+genid+"/", {}, function(info) {
        this.params.update(info.params);
    }.bind(this));
}

Sequence.prototype._make_name = function() {
    var gen = $("#sequence #seqgen option").filter(":selected").text()
    var txt = [];
    var d = new Date();
    var datestr =  d.getFullYear()+"."+(d.getMonth()+1)+"."+d.getDate()+" ";

    $("#sequence #seqparams input").each(function() { txt.push(this.name+"="+this.value); })
    return gen+":["+txt.join(", ")+"]"
}
Sequence.prototype.edit = function() {
    var _this = this;
    var curname = this._make_name();
    $("#sequence #seqlist").replaceWith("<input id='seqlist' name='seq_name' type='text' value='"+curname+"' />");
    $("#seqgen, #seqparams input, #seqstatic").removeAttr("disabled");
    var setname = function() { $("#seqlist").attr("value", _this._make_name()); };
    $("#sequence #seqgen").change(function() {
        setname();
        $("#sequence #seqparams input").bind("blur.setname", setname );
    });
    $("#sequence #seqparams input").bind("blur.setname", setname );
    $("#sequence #seqlist").blur(function() {
        if (this.value != _this._make_name())
            $("#sequence #seqparams input").unbind("blur.setname");
    })
}


Sequence.prototype.enable = function() {
    $("#seqlist").removeAttr("disabled");
}
Sequence.prototype.disable = function() {
    $("#seqlist").attr("disabled", "disabled");
    $("#seqgen").attr("disabled", "disabled");
    $("#seqparams input").attr("disabled", "disabled");
    $("#seqstatic").attr("disabled", "disabled");   
}
Sequence.prototype.get_data = function() {
    if ($("#sequence #seqlist").get(0).tagName == "INPUT") {
        //This is a new sequence, create new!
        var data = {};
        data['name'] = $("#seqlist").attr("value");
        data['generator'] = $("#seqgen").attr("value");
        data['params'] = {};
        $("#seqparams input").each(function() { data['params'][this.name] = this.value; });
        data['static'] = $("#seqstatic").attr("checked") == "checked";
        return data;
    }
    return $("#sequence #seqlist").attr("value");
}