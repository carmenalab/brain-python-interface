var te = null;
$(document).ready(function() {
	$("table#main tr").each(function() {
		var idx = this.id;
		if (this.id == "newentry") {
			$(this).click(function() {
				if (te) te.destroy();
				te = new TaskEntry(null);
			})
		} else {
			$(this).click(function() {
				if (te) te.destroy();
				te = new TaskEntry(idx);
			})
		}
	})
	btn = false;
	$("#copybtn").click(TaskEntry.copy);
	$("#experiment").submit(function() {
		te.run(btn);
		return false;
	})
	$("#startbtn").click(function() { btn = true; })
	$("#testbtn").click(function() { btn = false; })
})

function TaskEntry(idx, info){
	if (idx) {
		this.idx = parseInt(idx.match(/row(\d+)/)[1]);
		this.tr = $("#"+idx);
		$("#copybtn").show();
		$("#startbtn, #testbtn").hide();
		$.getJSON("ajax/exp_info/"+this.idx+"/", {}, function (expinfo) {
			this.set(expinfo);
			this.disable();
			$("#border").show("slide", "fast");
		}.bind(this));
	} else {
		this.idx = null;
		this.tr = $("#newentry").show();
		$("#tasks").change(this._task_query.bind(this));
		$("#features input").change(this._task_query.bind(this));
		$("#copybtn").hide();
		$("#startbtn, #testbtn").show();
		if (info) {
			this.set(info);
			this.enable();
			$("#border").show("slide", "fast");
		} else {
			this._task_query(function(params, seqs) {
				this.init(params, seqs);
				this.enable();
				$("#border").show("slide", "fast");
			}.bind(this));
		}
	}
	this.tr.unbind("click");
	this.tr.addClass("rowactive active");
}
TaskEntry.prototype.init = function(params, seqs) {
	this.params = new Parameters(params);
	$("#parameters").append(this.params.obj);
	this.sequence = new Sequence(seqs?seqs:{});
	
	if (seqs) {
		$("#sequence").show()
	} else {
		$("#sequence").hide()
	}
}
TaskEntry.prototype.set = function(info) {
	this.expinfo = info;
	$("#tasks option").each(function() {
		if (this.value == info.task)
			this.selected = true;
	})
	$("#subjects option").each(function() {
		if (this.value == info.subject)
			this.selected = true;
	});
	$("#features input[type=checkbox]").each(function() {
		this.checked = false;
		for (var idx in info.feats) {
			if (this.name == info.feats[idx])
				this.checked = true;
		}
	});
	this.init(info.params, info.sequence);
}
TaskEntry.copy = function() {
	te.destroy();
	te = new TaskEntry(null, te.expinfo);
}
TaskEntry.prototype.destroy = function() {
	$("#border").hide();
	this.tr.removeClass("rowactive");
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
		if (typeof(callback) == "function") {
			callback(taskinfo.params, taskinfo.sequence)
		} else {
			this.params.update(taskinfo.params);
			this.sequence.update(taskinfo.sequence);
			if (taskinfo.sequence)
				$("#sequence").show()
			else
				$("#sequence").hide()
		}
	}.bind(this));
}

TaskEntry.prototype.run = function(save) {
	this.disable();
	var form = {};
	form['csrfmiddlewaretoken'] = $("#experiment input").filter("[name=csrfmiddlewaretoken]").attr("value")
	form['data'] = JSON.stringify(this.get_data());
	$.post(save?"start":"test", form, function() {
		$("#testbtn").hide()
		$("#startbtn").attr("value", "Stop");
		$("#startbtn").unbind("click").click(this.stop.bind(this));
		$("#border").addClass("running");
		this.tr.addClass("running");
	}.bind(this));
}
TaskEntry.prototype.stop = function(data) {
	$("#border").removeClass("running");
	this.tr.removeClass("running");
	$("#content .startbtn").hide()
	$("#addbtn").show()
	$("#copybtn").show().attr("onclick", "startnew("+this.idx+")")
}

TaskEntry.prototype.get_data = function() {
	var data = {};
	data['subject'] = parseInt($("#subjects").attr("value"));
	data['task'] = parseInt($("#tasks").attr("value"));
	data['feats'] = {};
	$("#experiment #features input").each(function() {
		if (this.checked)
			data.feats[this.value] = this.name;
	})
	data['params'] = this.params.to_json();
	data['sequence'] = this.sequence.get_data();

	return data
}
TaskEntry.prototype.enable = function() {
	$("#parameters input, #features input").removeAttr("disabled");
	if (this.sequence)
		this.sequence.enable();
	if (this.idx)
		$("#subjects input, #tasks input").removeAttr("disabled");
}
TaskEntry.prototype.disable = function() {
	$("#parameters input, #features input").attr("disabled", "disabled");
	if (this.sequence)
		this.sequence.disable();
	if (this.idx)
		$("#subjects input, #tasks input").attr("disabled", "disabled");
}
