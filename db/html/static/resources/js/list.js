var entries = [];
$(document).ready(function() {
	var rows = $("table#main tbody>tr");
	for (var i = 0; i < rows.length; i++) {
		entries.push(new TaskEntry(rows[i].id));
	}
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
	for (var i in params) {
		$("#parameters ul").append(
			"<li title='"+params[i][0]+"'>\n"+
			"	<label class='traitname' for='"+i+"'>"+i+"</label>\n"+
			"	<input id='"+i+"' name='"+i+"' type='text' value='"+params[i][1]+"' />"+
			"<div class='clear'></div></li>");
	}
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
		//Delete this row entirely
		entries.pop();
		this.tr.remove()
		$("#addbtn").show()
	}
}

TaskEntry.prototype.populate = function() {
	var _this = this;
	$.getJSON("ajax/exp_info/"+this.idx, {}, function(data) {
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