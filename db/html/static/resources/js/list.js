var entries = [];
$(document).ready(function() {
	$("table#main tbody>tr").each(function() {
		entries.push(new TaskEntry(this.id));
	})
})
function TaskEntry(idx){
	if (idx == -1) {
		//This is a new row, need to set task name
		this.newentry = true
		this.task = $("#tasks option").filter(":selected").text()
		this.tr = $("#newentry")
		var _this = this;
		$("#tasks").change(function() {
			_this.task = $("#tasks option").filter(":selected").text()
			_this._query_params()
		})
	} else {
		this.newentry = false
		this.idx = parseInt(idx.match(/row(\d+)/)[1])
		this.tr = $("#"+idx)
	}
	this.active = false
	var _this = this
	this.tr.click(function() {
		if (!_this.active) {
			_this.activate(); 
		}
	})
}
TaskEntry.prototype._add_fieldset = function(name, where) {
	$("#content div."+where).append(
		"<fieldset id='"+name.toLowerCase()+"'>\n"+
			"<legend>"+name+"</legend>\n"+
		"</fieldset>");
}
TaskEntry.prototype._setup_params = function(params) {
	var html = "";
	for (var i in params) {
		html += "<li title='"+params[i][0]+"'>\n"+
			"	<label class='traitname' for='"+i+"'>"+i+"</label>\n"+
			"	<input id='"+i+"' name='"+i+"' type='text' value='"+params[i][1]+"' />"+
			"<div class='clear'></div></li>";
	}
	$("#parameters ul").append(html);
}
TaskEntry.prototype._query_params = function() {
	var data = {}
	$("#features input").filter(":checked").each(function(i) {
		data[$(this).attr("name")] = true
	})

	var _this = this;
	$.getJSON("ajax/task_params/"+this.task, data, function(data){
		$("#parameters ul").html("");
		_this._setup_params(data);
		if (!_this.active) {
			_this.active = true;
			$("#content").show("drop");
		}
	})
}

TaskEntry.prototype.deactivate = function() {
	this.tr.removeClass("rowactive");
	this.active = false;
	if (this.newentry) {
		//Delete this row entirely4
		entries.pop();
		this.tr.remove()
		$("#addbtn").show()
	}
}

TaskEntry.prototype.populate = function() {
	var _this = this;
	$.getJSON("ajax/exp_info/"+this.idx, {}, function(data) {
		_this.sequence = new SequenceEditor(data['seqid'])
		_this._setup_params(data['params']);
		$("#notes textarea").html(data['notes']);
		for (var name in data['features']) {
			if (data['features'][name]) {
				$("#features input#"+name).attr("checked", "checked");
			}
		}
		if (!_this.active) {
			$("#content").show("drop");
			_this.active = true;
		}
		$("#content input,select").attr("disabled", true)
	})
}




function SequenceEditor(idx) {
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
	
	var _this = this;
	$("#sequence #seqgen").change(function() { _this._update_params(); });

	this.idx = idx
	if (typeof(idx) == "undefined") {
		$("#tasks").change(function() { _this._query_sequences($(this).attr("value")); });
		this._query_sequences($("#tasks").attr("value"));
	} else {
		this._query_sequences(idx, false);
	}
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
SequenceEditor.prototype._query_sequences = function(task, editable) {
	var _this = this;
	$.getJSON("ajax/task_seq/"+task, {}, function(data) {
		$("#sequence #seqlist").replaceWith("<select id='seqlist' name='seq_name'></select>");
		var html = "";
		for (var i in data) 
			html += "<option value='"+i+"'>"+data[i]+"</option>";
		$("#sequence #seqlist").append(html+"<option value='new'>Create New...</option>");

		if (typeof(editable) == "undefined")
			$("#sequence #seqlist").change(function() { _this._query_data(); });
		else
			$("#sequence #seqlist").attr("disabled", "disabled");
		
		_this._query_data(editable);
	})
}
SequenceEditor.prototype._query_data = function(editable) {
	var data_id = $("#sequence #seqlist").attr("value");

	var _this = this;
	if (data_id == "new") {
		this._update_params();
		this.edit();
	}
	else
		$.getJSON("ajax/seq_data/"+data_id, {}, function(data) { _this.set_data(data, editable); })
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

SequenceEditor.prototype.set_data = function(data, editable) {
	//Setup generator
	$("#sequence #seqgen option").filter(function() {
		return $(this).attr("value") == data['genid'];
	}).attr("selected", "selected")
	//Setup parameters
	this._update_params()
	for (var i in data['params']) {
		$("#sequence #seq_"+i).attr("value", data['params'][i])
	}
	//Setup static
	if (data['static'])
		$("#seqstatic").attr("checked", "checked")

	//Disable all the inputs
	$("#seqgen, #seqparams input, #seqstatic").attr("disabled", "disabled");
	//Only allow editing on new entries
	if (typeof(editable) == "undefined" || editable) {
		var _this = this;
		//Add a callback to enable editing
		$("#sequence #seqparams").bind("click.edit", function() { 
			$("#sequence #seqparams").unbind("click.edit");
			_this.edit(); 
		})
	}
}