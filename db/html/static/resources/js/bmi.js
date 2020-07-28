//////////////////// BMI ////////////////////////
/////////////////////////////////////////////////
/*
Runs the code for any button pushing on the BMI training sub-GUI
*/
var goodcells = /^\s*(?:good|bmi)?\s?(?:unit|cell)s?\s?[:-]\s*\n*(.*)$/gim;
var cellnames = /(\d{1,3})\s?(\w{1})/gim;
var parsecell = /(\d{1,3})\s?(\w{1})/;
var parsetime = /^(?:(\d{0,2}):)?(\d{1,2}):(\d{0,2}\.?\d*)$/;

function BMI(idx, info, notes) {
    this.idx = idx;
    try {
        this.cells = goodcells.exec(notes)[1].match(cellnames);
    } catch (e) {
        this.cells = [];
    }

    // Clear the 'Available' and 'Selected' cells fields of any stale options/selections
    $('#cells').empty()
    $('#available').empty()

    this.info = info;
    this.neuralinfo = info['_neuralinfo'];
    delete info['_neuralinfo']

    this.available = {};
    this.selected = {};

    if (this.neuralinfo !== null) {
        if (this.neuralinfo.is_seed) {
            for (var i = 0; i < this.neuralinfo.units.length; i++) {
                var name = this.neuralinfo.units[i][0];
                name += String.fromCharCode(this.neuralinfo.units[i][1]+96);
                this.remove(name);
            }

            this._bindui();
            this.cancel();
        }
    }
}
BMI.zeroPad = function(number, width) {
    width -= number.toString().length;
    if ( width > 0 ) {
        var w = /\./.test(number) ? 2 : 1;
        return new Array(width + w).join('0') + number;
    }
    return number + "";
}
BMI.hms = function(sec) {
    var h = Math.floor(sec / 60 / 60);
    var m = Math.floor(sec / 60 % 60);
    var s = Math.round(sec % 60);
    if (h > 0)
        return h+':'+BMI.zeroPad(m,2)+':'+BMI.zeroPad(s,2);
    return m+':'+BMI.zeroPad(s,2);
}
BMI.ptime = function(t) {
    var chunks = parsetime.exec(t);
    try {
        var secs = parseInt(chunks[2]) * 60;
        secs += parseFloat(chunks[3]);
        if (chunks[1] !== undefined)
            secs += parseInt(chunks[1]) * 60 * 60;

        return secs;
    } catch (e) {
        return null;
    }
}

/**
 * Move an item from one select field 'src' to another select field 'dst'
 * @param {string} name The name of the item
 * @param {object} src The associative array/object the item 'name' is currently in
 * @param {object} dst The associative array/object the item 'name' should be in
 * @param {string} dstname Name of the destination 'option' field
*/
BMI.swap = function(name, src, dst, dstname) {
    if (dst[name] === undefined) {
        var obj;
        if (src[name] !== undefined) {
            obj = src[name].remove()
            delete src[name];
        } else {
            // the option is not in the source menu, so make a new one
            obj = $("<option>"+name+"</option>");
        }

        var names = [name];
        for (var n in dst)
            names.push(n)
        dst[name] = obj;

        if (names.length == 1)
            return $('#'+dstname).append(obj);

        //Rudimentary lexical sort
        names.sort(function(b, a) {
            var an = parsecell.exec(a);
            var bn = parsecell.exec(b);
            var chan = parseInt(bn[1], 10) - parseInt(an[1], 10);
            var chr = (bn[2].charCodeAt(0) - an[2].charCodeAt(0));
            return chan + chr / 10;
        });
        var idx = names.indexOf(name);
        if (idx == 0)
            dst[names[idx+1]].before(obj);
        else
            dst[names[idx-1]].after(obj);
    }
}
BMI.prototype.add = function(name) {
    BMI.swap(name, this.available, this.selected, 'cells');
}
BMI.prototype.remove = function(name) {
    BMI.swap(name, this.selected, this.available, 'available');
}
BMI.prototype.parse = function(cells, addAvail) {
    var names = cells.match(cellnames);
    if (addAvail) {
        for (var i = 0, il = names.length; i < il; i++) {
            this.remove(names[i]);
        }
    } else {
        $("#cells option").each(function(idx, obj) {
            this.remove($(obj).text());
        }.bind(this));

        for (var i = 0, il = names.length; i < il; i++) {
            this.add(names[i]);
        }
        this.update();
    }
}
BMI.prototype.update = function(names) {
    var names = [];
    $('#cells option').each(function(idx, val) {
        names.push($(val).text());
    })
    $('#cellnames').val(names.join(', '));
}

BMI.prototype.set = function(name) {
    var info = this.info[name];
    $("#cells option").each(function(idx, obj) {
        this.remove($(obj).text());
    }.bind(this));

    for (var i = 0; i < info.units.length; i++) {
        var unit = info.units[i];
        var n = unit[0] + String.fromCharCode(unit[1]+96);
        this.add(n);
    }
    this.update();

    $("#bmibinlen").val(info.binlen);
    $("#tstart").val(BMI.hms(info.tslice[0]));
    $("#tend").val(BMI.hms(info.tslice[1]));
    $("#tslider").slider("values", info.tslice);
    $("#bmiclass option").each(function(idx, obj) {
        if ($(obj).text() == info.cls)
            $(obj).attr("selected", "selected");
    });
    $("#bmiextractor option:first").attr("selected", "selected");
}
BMI.prototype.new = function() {
    $("#cells option").each(function(idx, obj) {
        this.remove($(obj).text());
    }.bind(this));
    $("#bmibinlen").val("0.1");
    $("#bminame").replaceWith("<input id='bminame'>");
    //var selected_bmi_class = $("#bmiclass");    
    
    var selected_bmi_class = document.getElementById("bmiclass");
    var strSel = '_'.concat(selected_bmi_class.options[selected_bmi_class.selectedIndex].text);
    
    var new_bmi_name = this.neuralinfo.name.concat(strSel);
    $("#bminame").val(new_bmi_name);
    for (var i = 0; i < this.cells.length; i++) 
        this.add(this.cells[i]);
    $(".bmibtn").show();
    $("#bmi input,select,textarea").attr("disabled", null);
    this.update();
}
BMI.prototype.cancel = function() {
    $("#bmi input,select,textarea").attr("disabled", "disabled");
    $("#bminame").replaceWith("<select id='bminame' />");
    var i = 0;
    for (var name in this.info) {
        $("#bminame").append('<option>'+name+'</option>');
        i++;
    }

    if (i < 1)
        return this.new();

    $(".bmibtn").hide();
    $("#bminame").append("<option value='new'>Create New</option>");
    this.set($("#bminame option:first").text());

    var _this = this;
    $("#bminame").change(function(e){
        if (this.value == 'new')
            _this.new();
        else
            _this.set(this.value);
    })
}
BMI.prototype._bindui = function() {
    $("#tslider").slider({
        range:true, min:0, max:this.neuralinfo.length, values:[0, this.neuralinfo.length],
        slide: function(event, ui) {
            $("#tstart").val(BMI.hms(ui.values[0]));
            $("#tend").val(BMI.hms(ui.values[1]));
        },
    });
    $("#tstart").val(BMI.hms(0));
    $("#tend").val(BMI.hms(this.neuralinfo.length));
    $("#tstart").keyup(function(e) {
        var values = $("#tslider").slider("values");
        var sec = BMI.ptime(this.value);
        if (sec !== null) {
            $("#tslider").slider("values", [sec, values[1]]);
        }
        if (e.which == 13)
            this.value = BMI.hms(sec);
    });
    $("#tend").keyup(function(e) {
        var values = $("#tslider").slider("values");
        var sec = BMI.ptime(this.value);
        if (sec !== null) {
            $("#tslider").slider("values", [values[0], sec]);
        }
        if (e.which == 13)
            this.value = BMI.hms(sec);
    });
    $("#tstart").blur(function() {
        var values = $("#tslider").slider("values");
        this.value = BMI.hms(values[0]);
    });
    $("#tend").blur(function() {
        var values = $("#tslider").slider("values");
        this.value = BMI.hms(values[1]);
    });

    $('#makecell').click(function() {
        var units = $('#available option:selected');
        units.each(function(idx, obj) {
            this.add($(obj).text());
        }.bind(this));
        this.update();
    }.bind(this));

    $('#makeavail').click(function() {
        var units = $('#cells option:selected');
        units.each(function(idx, obj) {
            this.remove($(obj).text());
        }.bind(this));
        this.update();
    }.bind(this));

    $("#cellnames").blur(function(e) {
        console.log($("#cellnames").val())
        this.parse($("#cellnames").val());
    }.bind(this));

    $("#bmitrain").click(this.train.bind(this));
    $("#bmicancel").click(this.cancel.bind(this));
    $("#bmi").show();
}

// Destructor for the BMI sub-menu
BMI.prototype.destroy = function() {
    if (this.neuralinfo !== null) {
        if (this.neuralinfo.is_seed) {
            $("#tslider").slider("destroy");
            $("#tstart").unbind("keyup");
            $("#tstart").unbind("blur");
            $("#tend").unbind("keyup");
            $("#tend").unbind("blur");
            $("#makecell").unbind("click");
            $("#makeavail").unbind("click");
            $("#cellnames").unbind("click");
            $("#cellnames").unbind("blur");
            $("#bmitrain").unbind("click");
            $("#bmicancel").unbind("click");
            $("#bmi").hide();
        }
    }
}

BMI.prototype.train = function() {
    this.update();
    var csrf = $("#experiment input[name=csrfmiddlewaretoken]");
    var data = {};
    data.bminame = $("#bminame").val();
    data.bmiclass = $("#bmiclass").val();
    data.bmiextractor = $("#bmiextractor").val();
    data.cells = $("#cellnames").val();
    data.channels = $("#channelnames").val();
    data.bmiupdaterate = $("#bmiupdaterate").val();
    data.tslice = $("#tslider").slider("values");
    data.ssm = $("#ssm").val();
    data.pos_key = $("#pos_key").val();
    data.kin_extractor = $("#kin_extractor").val();
    data.zscore = $("#zscore").val();

    data.csrfmiddlewaretoken = csrf.val();

    $.post("/make_bmi/"+this.idx, data, function(resp) {
        if (resp.status == "success") {
            alert("BMI Training queued");
            this.cancel();
        } else
            alert(resp.msg);
    }.bind(this), "json");
}

