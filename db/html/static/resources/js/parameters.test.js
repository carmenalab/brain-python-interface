QUnit.test("Parameters creation", function( assert ) {
	  var p = new Parameters();
  	// assert.ok(p.obj.innerHTML == "", "Parameters table starts empty");

  	// add one of each type to the parameters
  	data = {"param1": {"type": "Float", "value": 1.0, "desc":"Parameter 1"},
  		"param2" : {"type": "Int", "value": 1, "desc":"Parameter 2"},
  		"param3" : {"type": "Tuple", "value": [2, 2], "desc": "Tuple parameter", "default":[2, 2]},
  		"param4" : {"type": "Array", "value": [2, 2, 2], "desc": "Array parameter", "default":[2, 2, 2]},
  		"param5" : {"type": "Instance", "value": 3, "options": [[1, "value 1"], [2, "value 2"], [3, "value 3"]], "desc": "Instance"},
  		"param6" : {"type": "InstanceFromDB", "value": 2, "options": [[1, "value 1"], [2, "value 2"], [3, "value 3"]], "desc": "Instance"},
  		"param7" : {"type": "DataFile", "value": 1, "options": [[1, "value 1"], [2, "value 2"], [3, "value 3"]], "desc": "Instance"},
  		"param8" : {"type": "String", "value": "1.0", "desc": "String parameter"},
  		"param9" : {"type": "Enum", "value": "A", "desc": "Enum parameter", "options":["B", "A", "C"]},
  		"param10" : {"type": "OptionsList", "value": "A", "desc": "OptionsList parameter", "options":["B", "A", "C"]},
  		"param11": {"type": "Bool", "value": 1.0, "desc": "Boolean parameter"},
  	}
  	p.update(data);

  	
  	var k = 0;
  	for (var param in data) {
  		assert.equal(p.obj.rows[k].title, data[param]["desc"], "Parameter title " + k + " matches");
  		assert.equal(p.obj.rows[k].cells[1].firstChild.id, "param_param" + (k +1) )
  		assert.equal(p.obj.rows[k].cells[0].firstChild.innerHTML, param)

  		k += 1;
  	}
  	$("#seqparams").append(p.obj);

    // test that placeholder values also get grabbed
    data = {"param1": {"type":"Float", "value":"", "desc":"Parameter 1", "default":"10"}};
    p.update(data);
    param_data = p.to_json();
    assert.equal(Object.keys(param_data).length, 1)
    assert.equal(param_data["param1"], "10");

    // test that placeholder values get overwritten if a value is supplied
    data = {"param1": {"type":"Float", "value":"11", "desc":"Parameter 1", "default":"10"}};
    p.update(data);
    param_data = p.to_json();
    assert.equal(Object.keys(param_data).length, 1)
    assert.equal(param_data["param1"], "11");

    // cleanup
    $(p.obj).remove();  
    delete p; 
});