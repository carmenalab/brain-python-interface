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
	var _this = this;
	if (idx) {
		this.idx = parseInt(idx.match(/row(\d+)/)[1]);
		this.tr = $("#"+idx);
		$.getJSON("ajax/exp_info/"+idx, {}, function (expinfo) {
			_this.populate(info);
			$("#content").animate
		});
	} else {
		//This is a new row, need to set task name
		this.tr = $("#newentry").show();
		$("#tasks").change(function() {
			_this._query_params($("#tasks option").filter(":selected").text());
		})
		this._query_params($("#tasks option").filter(":selected").text());
	}
}
TaskEntry.prototype.populate = function(info) {

}
TaskEntry.prototype._query_params = function(task) {

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
TaskEntry.prototype.enable = function() {
	$("#parameters input, #features input").removeAttr("disabled");
	this.sequence.enable();
	if (this.newentry)
		$("#subjects input, #tasks input").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
	$("#parameters input, #features input").attr("disabled", "disabled");
	this.sequence.disable();
	if (this.newentry)
		$("#subjects input, #tasks input").attr("disabled", "disabled");
}
