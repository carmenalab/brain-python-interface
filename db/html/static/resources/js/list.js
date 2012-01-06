var entries = [];
$(document).ready(function() {
	$("table#main tbody>tr").each(function() {
		entries.push(new TaskEntry(this.id));
	})
})

function start_experiment() {
	var data = new Object();
	data['subject_id'] = $("#subjects").attr("value");
	data['task_id'] = $("#tasks").attr("value");
	data['feats'] = new Array();
	$("#experiment #features input").each(function() {
		if ($(this).attr("checked"))
			data['feats'].push(this.name);
	})
	data['params'] = new Object();
	$("#experiment #parameters input").each(function() {
		data['params'][this.name] = this.value;
	})
	data['sequence'] = entries[entries.length-1].sequence.get_data();

	var form = new Object();
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(data);

	$.post("start", form, function(data) { 
		$("#newentry td.colDate").html(data['date']);
		$("#newentry td.colSubj").html(data['subj']);
		$("#newentry td.colTask").html(data['task']);
		$("#newentry").attr("id", "row"+data['id']);
		entries[entries.length-1].idx = data['id'];
		entries[entries.length-1]._runstart(); 
	})
}
function stop_experiment() {
	$.getJSON("stop", {}, function(data) {
		if (data == "success") {
			for (var i in entries)
				if (entries[i].running)
					break;
			entries[i]._runstop()
		}
	})
}

function TaskEntry(idx){
	if (idx == -1) {
		//This is a new row, need to set task name
		this.newentry = true
		this.tr = $("#newentry")
		var _this = this;
		$("#tasks").change(function() {
			_this._query_params($("#tasks option").filter(":selected").text())
		})
	} else {
		this.newentry = false
		this.idx = parseInt(idx.match(/row(\d+)/)[1])
		this.tr = $("#"+idx)
	}
	this.active = false;
	this.running = this.tr.hasClass("running");
	var _this = this;
	this.tr.click(function() {
		if (!_this.active) {
			_this.activate();
		}
	})
	if (this.running) {
		this.activate();
		$("#addbtn").hide()
	}

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
			"	<input id='"+i+"' name='"+i+"' type='text' value='"+JSON.stringify(params[i][1])+"' />"+
			"<div class='clear'></div></li>";
	}
	$("#parameters ul").append(html);
}
TaskEntry.prototype._query_params = function(task) {
	var data = {}
	$("#features input").filter(":checked").each(function(i) {
		data[$(this).attr("name")] = true
	})

	var _this = this;
	$.getJSON("ajax/task_params/"+task, data, function(data){
		$("#parameters ul").html("");
		_this._setup_params(data);
		if (!_this.active) {
			_this.active = true;
			$("#content").show("drop");
		}
	})
}
TaskEntry.prototype._runstart = function(data) {
	this.newentry = false;
	this.running = true;
	this.disable();
	
	$("#content input[type='submit']").attr("value", "Stop Experiment");
	$("#content form").attr("action", "javascript:stop_experiment();");
	$("#content").addClass("running");
	this.tr.addClass("running");
}
TaskEntry.prototype._runstop = function(data) {
	this.running = false;
	$("#content").removeClass("running");
	this.tr.removeClass("running");
	$("#content input[type='submit']").hide()
	$("#content form").removeAttr("action");
}

TaskEntry.prototype.deactivate = function() {
	this.tr.removeClass("active rowactive");
	$("#content").removeClass("running");
	this.active = false;
	if (this.newentry) {
		//Delete this row entirely
		entries.pop();
		this.tr.remove()
		$("#addbtn").show()
	}
}
TaskEntry.prototype.disable = function() {
	$("#parameters input, #features input").attr("disabled", "disabled");
	this.sequence.disable();
	if (this.newentry)
		$("#subjects input, #tasks input").attr("disabled", "disabled");
}
TaskEntry.prototype.populate = function(idx, disable) {
	var _this = this;
	$.getJSON("ajax/exp_info/"+idx, {}, function(data) {
		_this.sequence = new SequenceEditor(data['task'], data['seqid']);
		_this._setup_params(data['params']);
		$("#notes textarea").html(data['notes']);
		for (var name in data['features']) {
			if (data['features'][name]) {
				$("#features input#"+name).attr("checked", "checked");
			}
		}
		if (typeof(disable) == "undefined" || disable)
			_this.disable();
		if (!_this.active) {
			$("#content").show("drop");
			_this.active = true;
		}
	})
}




function SequenceEditor(task, idx) {
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
		this._query_sequences(task)
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
SequenceEditor.prototype._query_sequences = function(task) {
	var _this = this;
	$.getJSON("ajax/task_seq/"+task, {}, function(data) {
		$("#sequence #seqlist").replaceWith("<select id='seqlist' name='seq_name'></select>");
		var html = "";
		for (var i in data) 
			html += "<option value='"+i+"'>"+data[i]+"</option>";
		$("#sequence #seqlist").append(html+"<option value='new'>Create New...</option>");
		$("#sequence #seqlist").change(function() { _this._query_data(); });
		
		if (typeof(_this.idx) != "undefined") {
			$("#sequence #seqlist").attr("disabled", "disabled");
			$("#sequence #seqlist option").each(function() {
				if (this.value == _this.idx)
					$(this).attr("selected", "selected");
			})
			_this._query_data(false);
		} else
			_this._query_data(true);
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
		$("#sequence #seq_"+i).attr("value", JSON.stringify(data['params'][i]))
	}
	//Setup static
	if (data['static'])
		$("#seqstatic").attr("checked", "checked")

	//Disable all the inputs
	this.disable();
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
SequenceEditor.prototype.disable = function() {
	$("#seqgen, #seqparams input, #seqstatic").attr("disabled", "disabled");
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


function Report(update, data) {
	this.obj = $("#content #report");

	var html = "";
	if (typeof(update) != "undefined" && update ) {
		html += "<label for='id"
	} else {
		
	}
}