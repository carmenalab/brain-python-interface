// //import $ from "jquery";
// var $ = require('jquery');
// const list = require("./list.js")


// test('Sequence object can be created', () => {
//     var seq = new list.Sequence();
//     expect(seq.options).toEqual({});

//     seq.destroy();
// });


// // Let's test this function
// function isEven(val) {
//     return val % 2 === 0;
// }
 
// test('isEven()', function() {
//     ok(isEven(0), 'Zero is an even number');
//     ok(isEven(2), 'So is two');
//     ok(isEven(-4), 'So is negative four');
//     ok(!isEven(1), 'One is not an even number');
//     ok(!isEven(-7), 'Neither is negative seven');
// })




// QUnit.module("Faking response data", {
//     setup: function () {
//         var testData = { foo: 'bar', name: 'phil' };
//         this.server = sinon.fakeServer.create();
//         this.server.respondWith("GET", "/ajax/gen_info/gen1/", [200, { "Content-Type": "application/json" }, JSON.stringify(testData)]);
//         this.server.respondWith("GET", "/ajax/gen_info/gen2/", [200, { "Content-Type": "application/json" }, JSON.stringify(testData)]);
//     },
//     teardown: function () {
//         this.server.restore();
//     }
// });





function isEmpty(object) {
	for(var i in object) { 
		return false; 
	} 
	return true; 
} 

QUnit.test("Sequence creation", function( assert ) {
	var seq = new Sequence();
  	assert.ok(isEmpty(seq.options), "Passed!" );
});



QUnit.test("ajax tests", function (assert) {
    var xhr = sinon.useFakeXMLHttpRequest();
    var requests = sinon.requests = [];

    xhr.onCreate = function (request) {
        requests.push(request);
    };

    // var callback = sinon.spy();
    function callback(x) {
		console.log(x)
	}

    // $.getJSON('/ajax/gen_info/gen1/', {}, callback);
    var seq = new Sequence();
    seq.value = "gen1" // TODO HOW IS THIS SET IN THE CODE?!
    seq._handle_chgen();

    assert.equal(sinon.requests.length, 1);
    assert.equal(sinon.requests[0].url, "/ajax/gen_info/gen1/");

    requests[0].respond(200, { "Content-Type": "application/json" }, 
'{"param1":1, "param2": "3"}');

    // assert.ok(callback.called, "stuff");
});

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
});

