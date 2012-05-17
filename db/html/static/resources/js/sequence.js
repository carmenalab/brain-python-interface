function SequenceEditor(task, idx, editable) {
    var html = "<legend>Sequence</legend>";
    html += "<label class='traitname' for='seqlist'>Name:</label>";
    html += "<select id='seqlist' name='seq_name'></select><div class='clear'></div>";
    html += "<label class='traitname' for='seqgen'>Generator:</label>";
    html += "<select id='seqgen' name='seq_gen'></select><div class='clear'></div>";
    html += "<div id='seqparams'></div>";
    html += "<div id='seqstatic_div'><input id='seqstatic' type='checkbox' name='static' />";
    html += "<label for='seqstatic'>Static</label></div>"
    $("#sequence").html(html);
    for (var i in genparams)
        $("#sequence #seqgen").append("<option value='"+i+"'>"+genparams[i][0]+"</option>");
}


SequenceEditor.prototype._update_params = function() {
    var idx = $("#sequence #seqgen").attr("value");
    var params = genparams[idx][1].split(",")

    //If generator is not static, add a length parameter
    if (genparams[idx][2]) {
        params.unshift("length");
        $("#seqstatic_div").show()
    } else {
        $("#seqstatic_div").hide()
    }

    var html = "";
    for (var i in params) {
        html += "<label class='traitname' for='seq_"+params[i]+"'>"+params[i]+"</label>";
        html += "<input id='seq_"+params[i]+"' type='text' name='"+params[i]+"' />";
        html += "<div class='clear'></div>";
    }
    $("#sequence #seqparams").html(html)
}
SequenceEditor.prototype._query_sequences = function(task) {
    var _this = this;
    $.getJSON("ajax/task_seq/"+task, {}, function(data) {
        $("#sequence #seqlist").replaceWith("<select id='seqlist' name='seq_name'></select>");
        var html = "";
        for (var i in data) 
            html += "<option value='"+i+"'>"+data[i]+"</option>";
        $("#sequence #seqlist").append(html+"<option value='new'>Create New...</option>");
        $("#sequence #seqlist").change(function() { _this._query_data(); });
        
        if (!_this.editable)
            $("#sequence #seqlist").attr("disabled", "disabled");
        console.log($("#seqlist").attr("disabled"))
        
        if (typeof(_this.idx) != "undefined") {
            $("#sequence #seqlist option").each(function() {
                if (this.value == _this.idx)
                    $(this).attr("selected", "selected");
            })
        }
        _this._query_data();
    })
}
SequenceEditor.prototype._make_name = function() {
    var gen = $("#sequence #seqgen option").filter(":selected").text()
    var txt = [];
    var d = new Date();
    var datestr =  d.getFullYear()+"."+(d.getMonth()+1)+"."+d.getDate()+" ";

    $("#sequence #seqparams input").each(function() { txt.push(this.name+"="+this.value); })

    return gen+":["+txt.join(", ")+"]"
}
SequenceEditor.prototype.edit = function() {
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

SequenceEditor.prototype.set_data = function(data) {
    //Setup generator
    $("#sequence #seqgen option").filter(function() {
        return $(this).attr("value") == data['genid'];
    }).attr("selected", "selected");
    //Setup parameters
    this._update_params();
    for (var i in data['params']) {
        $("#sequence #seq_"+i).attr("value", JSON.stringify(data['params'][i]))
    }
    //Setup static
    if (data['static'])
        $("#seqstatic").attr("checked", "checked")

    //Disable all the inputs
    $("#seqgen, #seqparams input, #seqstatic").attr("disabled", "disabled");
    $("#seqparams input").attr("disabled", "disabled");
    $("#seqstatic").attr("disabled", "disabled");
    //Only allow editing on new entries
    if (this.editable) {
        var _this = this;
        //Add a callback to enable editing
        $("#sequence #seqparams").bind("click.edit", function() { 
            $("#sequence #seqparams").unbind("click.edit");
            _this.edit(); 
        })
    }
}
SequenceEditor.prototype.enable = function() {
    $("#seqlist").removeAttr("disabled");
}
SequenceEditor.prototype.disable = function() {
    $("#seqlist").attr("disabled", "disabled");
    $("#seqgen").attr("disabled", "disabled");
    $("#seqparams input").attr("disabled", "disabled");
    $("#seqstatic").attr("disabled", "disabled");   
}
SequenceEditor.prototype.get_data = function() {
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