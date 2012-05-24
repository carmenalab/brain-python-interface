var te = null;
$(document).ready(function() {
	$("table#main tr").each(function() {
		var idx = this.id;
		if (this.id == "newentry")
			idx = null;
		$(this).click(function() {
			if (te) te.destroy();
			te = new TaskEntry(idx);
		})
	})
})

function start_experiment() {
	var form = new Object();
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(entries[entries.length-1].get_data());

	$.post("start", form, function(data) { 
		$("#newentry td.colDate").html(data['date']);
		$("#newentry td.colSubj").html(data['subj']);
		$("#newentry td.colTask").html(data['task']);
		$("#newentry").attr("id", "row"+data['id']);
		entries[entries.length-1].idx = data['id'];
		entries[entries.length-1]._runstart(); 
		entries[entries.length-1].newentry = false;
	})
}
function test_experiment() {
	var form = new Object();
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(entries[entries.length-1].get_data());
	$(window).unload(function() {
		$.getJSON("stop", {}, function() {})
	});
	$.post("test", form, function(data) {
		entries[entries.length-1]._teststart();
	})
}
function stop_experiment() {
	$(window).unbind("unload");
	$.getJSON("stop", {}, function(data) {
		if (data == "running") {
			for (var i in entries)
				if (entries[i].running)
					break;
			entries[i]._runstop()
		} else if (data == "testing") {
			entries[entries.length-1]._teststop();
		}
	})
}

function TaskEntry(idx){
	var callback = (function(params, sequence) {
		this.sequence = new Sequence(sequence);
		this.params = new Parameters(params);
		$("#parameters").append(this.params.obj);
		$("#content").show("slide", "fast");
		if (this.idx)
			this.disable()
		else
			this.enable()
	}).bind(this);

	if (idx) {
		this.idx = parseInt(idx.match(/row(\d+)/)[1]);
		this.tr = $("#"+idx);
		$.getJSON("ajax/exp_info/"+this.idx+"/", {}, function (expinfo) {
			$("#features input[type=checkbox]").each(function() {
				this.checked = false;
				for (var idx in expinfo.feats) {
					if (this.name == expinfo.feats[idx])
						this.checked = true;
				}
			});
			callback(expinfo['params'], expinfo['seq']);
		}.bind(this));
	} else {
		//This is a new row, need to set task name
		this.idx = null;
		this.tr = $("#newentry").show();
		$("#tasks").change(this._task_query.bind(this));
		$("#features input").change(this._task_query.bind(this));
		this._task_query(callback);
	}
	this.tr.unbind("click");
	this.tr.addClass("rowactive active");
}
TaskEntry.prototype.destroy = function() {
	$("#content").hide();
	this.tr.removeClass("rowactive");
	$("#features input[type=checkbox]").each(function() {
		this.removeAttribute("checked");
	})
	try {
		$(this.params.obj).remove()
	} catch(e) {}
	this.tr.removeClass("rowactive active");
	this.sequence.destroy();
	delete this.params
	if (this.idx != null) {
		var idx = "row"+this.idx;
		this.tr.click(function() {
			if (te) te.destroy();
			te = new TaskEntry(idx);
		})
	} else {
		//Remove the newentry row
		this.tr.hide()
		//Rebind the click action
		this.tr.click(function() {
			if (te) te.destroy();
			te = new TaskEntry(null);
		})
		//Clean up event bindings
		$("#features input").unbind("change");
		$("#tasks").unbind("change");
	}
}
TaskEntry.prototype._task_query = function(callback) {
	var taskid = $("#tasks").attr("value");
	var feats = {};
	$("#features input").each(function() { 
		if (this.checked) 
			feats[this.name] = this.checked;	
	});

	$.getJSON("ajax/task_info/"+taskid+"/", feats, function(taskinfo) {
		if (typeof(callback) == "function")
			callback(taskinfo.params, taskinfo.sequences);
		else {
			this.params.update(taskinfo.params);
			this.sequence.update(taskinfo.sequences);
		}
	}.bind(this));
}

TaskEntry.prototype._runstart = function(data) {
	this.newentry = false;
	this.running = true;
	this.disable();
	
	$("#content #testbtn").hide()
	$("#content #startbtn").attr("value", "Stop Experiment");
	$("#content #startbtn").attr("onclick", "stop_experiment()");
	$("#content").addClass("running");
	this.tr.addClass("running");
}
TaskEntry.prototype._runstop = function(data) {
	this.running = false;
	$("#content").removeClass("running");
	this.tr.removeClass("running");
	$("#content .startbtn").hide()
	$("#addbtn").show()
	$("#copybtn").show().attr("onclick", "startnew("+this.idx+")")
}
TaskEntry.prototype._teststart = function(data) {
	this.running = true;
	this.disable();
	
	$("#content #testbtn").hide()
	$("#content #startbtn").attr("value", "Stop test");
	$("#content #startbtn").attr("onclick", "stop_experiment()");
	$("#content").addClass("running");
	this.tr.addClass("running");
}
TaskEntry.prototype._teststop = function(data) {
	this.running = false;
	this.enable(); 

	$("#content").removeClass("running");
	this.tr.removeClass("running");
	$("#content .startbtn").show()
	$("#content #startbtn").attr("value", "Start Experiment");
	$("#content #startbtn").attr("onclick", "start_experiment()");
}

TaskEntry.prototype.get_data = function() {
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

	return data
}
TaskEntry.prototype.enable = function() {
	$("#parameters input, #features input").removeAttr("disabled");
	this.sequence.enable();
	if (this.idx)
		$("#subjects input, #tasks input").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
	$("#parameters input, #features input").attr("disabled", "disabled");
	this.sequence.disable();
	if (this.idx)
		$("#subjects input, #tasks input").attr("disabled", "disabled");
}
