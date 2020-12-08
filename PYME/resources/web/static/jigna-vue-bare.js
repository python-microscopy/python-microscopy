(function (){

///// EventTarget /////////////////////////////////////////////////////////////
// Copyright (c) 2010 Nicholas C. Zakas. All rights reserved.
// MIT License
///////////////////////////////////////////////////////////////////////////////

var EventTarget = function(){
    this._listeners = {};
};

EventTarget.prototype = {

    constructor: EventTarget,

    add_listener: function(obj, event_name, listener, thisArg){
        var id = this._to_id(obj);

        if (this._listeners[id] === undefined){
            this._listeners[id] = {};
        }

        if (this._listeners[id][event_name] === undefined) {
            this._listeners[id][event_name] = [];
        }

        this._listeners[id][event_name].push({thisArg: thisArg, listener: listener});
    },

    fire_event: function(obj, event){
        var id = this._to_id(obj);

        if (typeof event == "string"){
            event = { name: event };
        }
        if (!event.target){
            event.target = obj;
        }

        if (!event.name){  //falsy
            throw new Error("Event object missing 'name' property.");
        }

        if (this._listeners[id] === undefined) {
            return;
        }

        if (this._listeners[id][event.name] instanceof Array){
            var listeners = this._listeners[id][event.name];
            for (var i=0, len=listeners.length; i < len; i++){
                listener = listeners[i].listener;
                thisArg = listeners[i].thisArg;
                listener.call(thisArg, event);
            }
        }
    },

    remove_listener: function(obj, event_name, listener){
        var id = this._to_id(obj);

        if (this._listeners[id][event_name] instanceof Array){
            var listeners = this._listeners[id][event_name];
            for (var i=0, len=listeners.length; i < len; i++){
                if (listeners[i] === listener){
                    listeners.splice(i, 1);
                    break;
                }
            }
        }
    },

    //// Private protocol /////////////////////////////////////////////////////

    _to_id: function(obj){
        if (obj.__id__ !== undefined) {
            return obj.__id__;
        }
        else {
            return obj;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////
// Jigna
///////////////////////////////////////////////////////////////////////////////

// Namespace for all Jigna-related objects.
var jigna = new EventTarget();

jigna.initialize = function(options) {
    options = options || {};
    this.ready  = $.Deferred();
    this.debug  = options.debug;
    this.async  = options.async;
    this.client = options.async ? new jigna.AsyncClient() : new jigna.Client();
    this.client.initialize();
    return this.ready;
};

jigna.models = {};

jigna.add_listener('jigna', 'model_added', function(event){
    var models = event.data;
    for (var model_name in models) {
        jigna.models[model_name] = models[model_name];
    }

    jigna.fire_event('jigna', 'object_changed');
});

jigna.threaded = function(obj, method_name, args) {
    args = Array.prototype.slice.call(arguments, 2);
    return this.client.call_instance_method_thread(obj.__id__, method_name, args);
};


///////////////////////////////////////////////////////////////////////////////
// Client
///////////////////////////////////////////////////////////////////////////////

jigna.Client = function() {};

jigna.Client.prototype.initialize = function() {
    // jigna.Client protocol.
    this.bridge           = this._get_bridge();

    // Private protocol.
    this._id_to_proxy_map = {};
    this._proxy_factory   = this._create_proxy_factory();

    // Add all of the models being edited
    jigna.add_listener(
        'jigna',
        'context_updated',
        function(event){this._add_models(event.data);},
        this
    );

    // Wait for the bridge to be ready, and when it is ready, update the
    // context so that initial models are added to jigna scope
    var client = this;
    this.bridge.ready.done(function(){
        client.update_context();
    });
};

jigna.Client.prototype.handle_event = function(jsonized_event) {
    /* Handle an event from the server. */
    var event = JSON.parse(jsonized_event);
    jigna.fire_event(event.obj, event);
};

jigna.Client.prototype.on_object_changed = function(event){
    if (jigna.debug) {
        this.print_JS_message('------------on_object_changed--------------');
        this.print_JS_message('object id  : ' + event.obj);
        this.print_JS_message('attribute  : ' + event.name);
        this.print_JS_message('items event: ' + event.items_event);
        this.print_JS_message('new type   : ' + event.data.type);
        this.print_JS_message('new value  : ' + event.data.value);
        this.print_JS_message('new info   : ' + event.data.info);
        this.print_JS_message('-------------------------------------------');
    }

    var proxy = this._id_to_proxy_map[event.obj];

    // If the *contents* of a list/dict have changed then we need to update
    // the associated proxy to reflect the change.
    if (event.items_event) {
        var collection_proxy = this._id_to_proxy_map[event.data.value];
        // The collection proxy can be undefined if on the Python side you
        // have re-initialized a list/dict with the same value that it
        // previously had, e.g.
        //
        // class Person(HasTraits):
        //     friends = List([1, 2, 3])
        //
        // fred = Person()
        // fred.friends = [1, 2, 3] # No trait changed event!!
        //
        // This is because even though traits does copy on assignment for
        // lists/dicts (and hence the new list will have a new Id), it fires
        // the trait change events only if it considers the old and new values
        // to be different (ie. if does not compare the identity of the lists).
        //
        // For us(!), it means that we won't have seen the new list before we
        // get an items changed event on it.
        if (collection_proxy === undefined) {
            proxy.__cache__[event.name] = this._create_proxy(
                event.data.type, event.data.value, event.data.info
            );

        } else {
            this._proxy_factory.update_proxy(
                collection_proxy, event.data.type, event.data.info
            );
        }

    } else {
        proxy.__cache__[event.name] = this._unmarshal(event.data);
    }

    // Angular listens to this event and forces a digest cycle which is how it
    // detects changes in its watchers.
    jigna.fire_event('jigna', {name: 'object_changed', object: proxy});
};

jigna.Client.prototype.send_request = function(request) {
    /* Send a request to the server and wait for (and return) the response. */

    var jsonized_request  = JSON.stringify(request);
    var jsonized_response = this.bridge.send_request(jsonized_request);

    return JSON.parse(jsonized_response).result;
};

// Convenience methods for each kind of request //////////////////////////////

jigna.Client.prototype.call_instance_method = function(id, method_name, args) {
    /* Call an instance method */

    var request = {
        kind        : 'call_instance_method',
        id          : id,
        method_name : method_name,
        args        : this._marshal_all(args)
    };

    var response = this.send_request(request);
    var result = this._unmarshal(response);

    return result;
};

jigna.Client.prototype.call_instance_method_thread = function(id, method_name, args) {
    /* Call an instance method in a thread. Useful if the method takes long to
    execute and you don't want to block the UI during that time.*/

    var request = {
        kind        : 'call_instance_method_thread',
        id          : id,
        method_name : method_name,
        args        : this._marshal_all(args),
    };

    // the response of a threaded request is a marshalled version of a python
    // future object. We attach 'done' and 'error' handlers on that object to
    // resolve/reject our own deferred.
    var response = this.send_request(request);
    var future_obj = this._unmarshal(response);

    var deferred = new $.Deferred();

    jigna.add_listener(future_obj, 'done', function(event){
        deferred.resolve(event.data);
    });

    jigna.add_listener(future_obj, 'error', function(event){
        deferred.reject(event.data);
    });

    return deferred.promise();
};

jigna.Client.prototype.get_attribute = function(proxy, attribute) {
    /* Get the specified attribute of the proxy from the server. */

    var request = this._create_request(proxy, attribute);

    var response = this.send_request(request);
    var result = this._unmarshal(response);

    return result;
};

jigna.Client.prototype.print_JS_message = function(message) {
    var request = {
        kind: 'print_JS_message',
        value: message
    };

    this.send_request(request);
};

jigna.Client.prototype.set_instance_attribute = function(id, attribute_name, value) {
    var request = {
        kind           : 'set_instance_attribute',
        id             : id,
        attribute_name : attribute_name,
        value          : this._marshal(value)
    };

    this.send_request(request);
};

jigna.Client.prototype.set_item = function(id, index, value) {
    var request = {
        kind  : 'set_item',
        id    : id,
        index : index,
        value : this._marshal(value)
    };

    this.send_request(request);
};

jigna.Client.prototype.update_context = function() {
    var request  = {kind : 'update_context'};

    this.send_request(request);
};

// Private protocol //////////////////////////////////////////////////////////

jigna.Client.prototype._add_model = function(model_name, id, info) {
    // Create a proxy for the object identified by the Id...
    var proxy = this._create_proxy('instance', id, info);

    // fire the event to let the UI toolkit know that a new model was added
    var data = {};
    data[model_name] = proxy;

    jigna.fire_event('jigna', {
        name: 'model_added',
        data: data,
    });

    return proxy;
};

jigna.Client.prototype._add_models = function(context) {
    var client = this;
    var models = {};
    $.each(context, function(model_name, model) {
        if (jigna.models[model_name] === undefined) {
            proxy = client._add_model(model_name, model.value, model.info);
            models[model_name] = proxy;
        }
    });

    // Resolve the jigna.ready deferred, at this point the initial set of
    // models are set.  For example vue.js can now use these data models to
    // create the initial Vue instance.
    jigna.ready.resolve();

    return models;
};

jigna.Client.prototype._create_proxy_factory = function() {
    return new jigna.ProxyFactory(this);
};

jigna.Client.prototype._create_proxy = function(type, obj, info) {
    if (type === 'primitive') {
        return obj;
    }
    else {
        var proxy = this._proxy_factory.create_proxy(type, obj, info);
        this._id_to_proxy_map[obj] = proxy;
        return proxy;
    }
};

jigna.Client.prototype._create_request = function(proxy, attribute) {
    /* Create the request object for getting the given attribute of the proxy. */

    var request;
    if (proxy.__type__ === 'instance') {
        request = {
            kind           : 'get_instance_attribute',
            id             : proxy.__id__,
            attribute_name : attribute
        };
    }
    else if ((proxy.__type__ === 'list') || (proxy.__type__ === 'dict')) {
        request = {
            kind  : 'get_item',
            id    : proxy.__id__,
            index : attribute
        };
    }
    return request;
};

jigna.Client.prototype._get_bridge = function() {
    var bridge, qt_bridge;

    // Are we using the intra-process Qt Bridge...
    qt_bridge = window['qt_bridge'];
    if (qt_bridge !== undefined) {
        bridge = new jigna.QtBridge(this, qt_bridge);
    // ... or the inter-process web bridge?
    } else {
        bridge = new jigna.WebBridge(this);
    }

    return bridge;
};

jigna.Client.prototype._marshal = function(obj) {
    var type, value;

    if (obj instanceof jigna.Proxy) {
        type  = obj.__type__;
        value = obj.__id__;

    } else {
        type  = 'primitive';
        value = obj;
    }

    return {'type' : type, 'value' : value};
};

jigna.Client.prototype._marshal_all = function(objs) {
    var index;

    for (index in objs) {
        objs[index] = this._marshal(objs[index]);
    }

    // For convenience, as we modify the array in-place.
    return objs;
};

jigna.Client.prototype._unmarshal = function(obj) {

    if (obj === null) {
        return null;
    }

    if (obj.type === 'primitive') {
        return obj.value;

    } else {
        value = this._id_to_proxy_map[obj.value];
        if (value === undefined) {
            return this._create_proxy(obj.type, obj.value, obj.info);
        }
        else {
            return value;
        }
    }
};


///////////////////////////////////////////////////////////////////////////////
// AsyncClient
///////////////////////////////////////////////////////////////////////////////

// Inherit AsyncClient from Client
// Source: MDN docs (https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Object/create)
jigna.AsyncClient = function() {};
jigna.AsyncClient.prototype = Object.create(jigna.Client.prototype);
jigna.AsyncClient.prototype.constructor = jigna.AsyncClient;

jigna.AsyncClient.prototype.send_request = function(request) {
    /* Send a request to the server and wait for (and return) the response. */

    var jsonized_request  = JSON.stringify(request);

    var deferred = new $.Deferred();
    this.bridge.send_request_async(jsonized_request).done(function(jsonized_response){
        deferred.resolve(JSON.parse(jsonized_response).result);
    });

    return deferred.promise();
};

jigna.AsyncClient.prototype.call_instance_method = function(id, method_name, args) {
    /* Calls an instance method. Do not use this to call any long running
    methods.

    Note: Since this is an async client, it won't block the UI even if you
    call a long running method here but the UI updates (progress bars etc)
    won't be available until the server completes running that method.
    */
    var request = {
        kind        : 'call_instance_method',
        id          : id,
        method_name : method_name,
        args        : this._marshal_all(args)
    };
    var client = this;

    var deferred = new $.Deferred();
    this.send_request(request).done(function(response){
        deferred.resolve(client._unmarshal(response));
    });

    return deferred.promise();
};

jigna.AsyncClient.prototype.call_instance_method_thread = function(id, method_name, args) {
    /* Calls an instance method in a thread on the server. Use this to call
    any long running method on the server otherwise you won't get any UI
    updates on the client.
    */
    var request = {
        kind        : 'call_instance_method_thread',
        id          : id,
        method_name : method_name,
        args        : this._marshal_all(args),
    };
    var client = this;

    // Note that this deferred is resolved when the method called in a thread
    // finishes, not when the request to call the method finishes.
    // This is done to make this similar to the sync client so that the users
    // can attach their handlers when the method is done.
    var deferred = new $.Deferred();

    this.send_request(request).done(function(response){

        var future_obj = client._unmarshal(response);
        // the response of a threaded request is a marshalled version of a python
        // future object. We attach 'done' and 'error' handlers on that object to
        // resolve/reject our own deferred.

        jigna.add_listener(future_obj, 'done', function(event){
            deferred.resolve(event.data);
        });

        jigna.add_listener(future_obj, 'error', function(event){
            deferred.reject(event.data);
        });

    });

    return deferred.promise();
};

jigna.AsyncClient.prototype.get_attribute = function(proxy, attribute) {
    /* Get the specified attribute of the proxy from the server. */
    var client = this;

    // start a new request only if a request for getting that attribute isn't
    // already sent
    if (proxy.__state__[attribute] != 'busy') {
        proxy.__state__[attribute] = 'busy';

        var request = this._create_request(proxy, attribute);
        this.send_request(request).done(function(response){
            // update the proxy cache
            proxy.__cache__[attribute] = client._unmarshal(response);

            // fire the object changed event to trigger fresh fetches from
            // the cache
            jigna.fire_event('jigna', {name: 'object_changed', object: proxy});

            // set the state as free again so that further fetches ask the
            // server again
            proxy.__state__[attribute] = undefined;

        });
    }

    return proxy.__cache__[attribute];
};

jigna.AsyncClient.prototype.on_object_changed = function(event){
    if (jigna.debug) {
        this.print_JS_message('------------on_object_changed--------------');
        this.print_JS_message('object id  : ' + event.obj);
        this.print_JS_message('attribute  : ' + event.name);
        this.print_JS_message('items event: ' + event.items_event);
        this.print_JS_message('new type   : ' + event.data.type);
        this.print_JS_message('new value  : ' + event.data.value);
        this.print_JS_message('new info   : ' + event.data.info);
        this.print_JS_message('-------------------------------------------');
    }

    var proxy = this._id_to_proxy_map[event.obj];

    // If the *contents* of a list/dict have changed then we need to update
    // the associated proxy to reflect the change.
    if (event.items_event) {
        var collection_proxy = this._id_to_proxy_map[event.data.value];
        // The collection proxy can be undefined if on the Python side you
        // have re-initialized a list/dict with the same value that it
        // previously had, e.g.
        //
        // class Person(HasTraits):
        //     friends = List([1, 2, 3])
        //
        // fred = Person()
        // fred.friends = [1, 2, 3] # No trait changed event!!
        //
        // This is because even though traits does copy on assignment for
        // lists/dicts (and hence the new list will have a new Id), it fires
        // the trait change events only if it considers the old and new values
        // to be different (ie. if does not compare the identity of the lists).
        //
        // For us(!), it means that we won't have seen the new list before we
        // get an items changed event on it.
        if (collection_proxy === undefined) {
            // In the async case, we do not create a new proxy instead we
            // update the id_to_proxy map and update the proxy with the
            // dict/list event info.
            collection_proxy = proxy.__cache__[event.name];
            this._id_to_proxy_map[event.data.value] = collection_proxy;
        }
        this._proxy_factory.update_proxy(
            collection_proxy, event.data.type, event.data.info
        );

    } else {
        proxy.__cache__[event.name] = this._unmarshal(event.data);
    }

    // Angular listens to this event and forces a digest cycle which is how it
    // detects changes in its watchers.
    jigna.fire_event('jigna', {name: 'object_changed', object: proxy});
};

// Private protocol //////////////////////////////////////////////////////////

jigna.AsyncClient.prototype._create_proxy_factory = function() {
    return new jigna.AsyncProxyFactory(this);
};


///////////////////////////////////////////////////////////////////////////////
// ProxyFactory
///////////////////////////////////////////////////////////////////////////////

jigna.ProxyFactory = function(client) {
    // Private protocol.
    this._client = client;

    // We create a constructor for each Python class and then create the
    // actual proxies from those.
    this._type_to_constructor_map = {};

    // Create a new instance constructor when a "new_type" event is fired.
    jigna.add_listener(
        'jigna',
        'new_type',
        function(event){this._create_instance_constructor(event.data);},
        this
    );

};

jigna.ProxyFactory.prototype.create_proxy = function(type, id, info) {
    /* Create a proxy for the given type, id and value. */

    var factory_method = this['_create_' + type + '_proxy'];
    if (factory_method === undefined) {
        throw 'cannot create proxy for: ' + type;
    }

    return factory_method.apply(this, [id, info]);
};

jigna.ProxyFactory.prototype.update_proxy = function(proxy, type, info) {
    /* Update the given proxy.
     *
     * This is only used for list and dict proxies when their items have
     * changed.
     */

    var factory_method = this['_update_' + type + '_proxy'];
    if (factory_method === undefined) {
        throw 'cannot update proxy for: ' + type;
    }

    return factory_method.apply(this, [proxy, info]);
};

// Private protocol ////////////////////////////////////////////////////////////

// Instance proxy creation /////////////////////////////////////////////////////

jigna.ProxyFactory.prototype._add_instance_method = function(proxy, method_name){
    proxy[method_name] = function() {
        // In here, 'this' refers to the proxy!
        var args = Array.prototype.slice.call(arguments);

        return this.__client__.call_instance_method(
            this.__id__, method_name, args
        );
    };
};

jigna.ProxyFactory.prototype._add_instance_attribute = function(proxy, attribute_name){
    var descriptor, get, set;

    get = function() {
        // In here, 'this' refers to the proxy!
        var value = this.__cache__[attribute_name];
        if (value === undefined) {
            value = this.__client__.get_attribute(this, attribute_name);
            this.__cache__[attribute_name] = value;
        }

        return value;
    };

    set = function(value) {
        // In here, 'this' refers to the proxy!
        //
        // If the proxy is for a 'HasTraits' instance then we don't need
        // to set the cached value here as the value will get updated when
        // we get the corresponding trait event. However, setting the value
        // here means that we can create jigna UIs for non-traits objects - it
        // just means we won't react to external changes to the model(s).
        this.__cache__[attribute_name] = value;
        this.__client__.set_instance_attribute(
            this.__id__, attribute_name, value
        );
    };

    descriptor = {enumerable:true, get:get, set:set, configurable:true};
    Object.defineProperty(proxy, attribute_name, descriptor);
};

jigna.ProxyFactory.prototype._add_instance_event = function(proxy, event_name){
    var descriptor, set;

    set = function(value) {
        this.__cache__[event_name] = value;
        this.__client__.set_instance_attribute(
            this.__id__, event_name, value
        );
    };

    descriptor = {enumerable:false, set:set, configurable: true};
    Object.defineProperty(proxy, event_name, descriptor);
};

jigna.ProxyFactory.prototype._create_instance_constructor = function(info) {
    var constructor = this._type_to_constructor_map[info.type_name];
    if (constructor !== undefined) {
        return constructor;
    }

    constructor = function(type, id, client) {
        jigna.Proxy.call(this, type, id, client);

        /* Listen for changes to the object that the proxy is a proxy for! */
        var index;
        var info = this.__info__;

        for (index in info.attribute_names) {
            jigna.add_listener(
                this,
                info.attribute_names[index],
                client.on_object_changed,
                client
            );
        }

        for (index in info.event_names) {
            jigna.add_listener(
                this,
                info.event_names[index],
                client.on_object_changed,
                client
            );
        }
    };

    // This is the standard way to set up protoype inheritance in JS.
    //
    // The line below says "when the function 'constructor' is called via the
    // 'new' operator, then set the prototype of the created object to the
    // given object".
    constructor.prototype = Object.create(jigna.Proxy.prototype);
    constructor.prototype.constructor = constructor;

    for (index in info.attribute_names) {
        this._add_instance_attribute(
            constructor.prototype, info.attribute_names[index]
        );
    }

    for (index in info.event_names) {
        this._add_instance_event(
            constructor.prototype, info.event_names[index]
        );
    }

    for (index in info.method_names) {
        this._add_instance_method(
            constructor.prototype, info.method_names[index]
        );
    }

    // The info is only sent to us once per type, and so we store it in the
    // prototype so that we can use it in the constructor to get the names
    // of any atttributes and events.
    Object.defineProperty(
        constructor.prototype, '__info__', {value : info}
    );

    // This property is not actually used by jigna itself. It is only there to
    // make it easy to see what the type of the server-side object is when
    // debugging the JS code in the web inspector.
    Object.defineProperty(
        constructor.prototype, '__type_name__', {value : info.type_name}
    );

    this._type_to_constructor_map[info.type_name] = constructor;

    return constructor;
}

jigna.ProxyFactory.prototype._create_instance_proxy = function(id, info) {
    var constructor, proxy;

    // We create a constructor for each Python class and then create the
    // actual proxies as from those.
    constructor = this._type_to_constructor_map[info.type_name];
    if (constructor === undefined) {
        constructor = this._create_instance_constructor(info);
    }

    return new constructor('instance', id, this._client);
};

// Dict proxy creation /////////////////////////////////////////////////////////

jigna.ProxyFactory.prototype._create_dict_proxy = function(id, info) {
    var proxy = new jigna.Proxy('dict', id, this._client);
    this._populate_dict_proxy(proxy, info);

    return proxy;
};

jigna.ProxyFactory.prototype._delete_dict_keys = function(proxy) {
    /* Delete all keys of a previously used dict proxy. */
    var index, keys;

    keys = Object.keys(proxy);
    for (index in keys) {
        delete proxy[keys[index]];
    }
};

jigna.ProxyFactory.prototype._populate_dict_proxy = function(proxy, info) {
    var index;

    for (index in info.keys) {
        this._add_item_attribute(proxy, info.keys[index]);
    }
};

jigna.ProxyFactory.prototype._update_dict_proxy = function(proxy, info) {
    proxy.__cache__ = {}
    this._delete_dict_keys(proxy);
    this._populate_dict_proxy(proxy, info);
};

// List proxy creation /////////////////////////////////////////////////////////

jigna.ProxyFactory.prototype._create_list_proxy = function(id, info) {
    var proxy = new jigna.ListProxy('list', id, this._client);
    this._populate_list_proxy(proxy, info);

    return proxy;
};

jigna.ProxyFactory.prototype._delete_list_items = function(proxy) {
    /* Delete all items of a previously used list proxy. */

    for (var index=proxy.length-1; index >= 0; index--) {
        delete proxy[index];
    }
};

jigna.ProxyFactory.prototype._populate_list_proxy = function(proxy, info) {
    /* Populate the items in a list proxy. */

    for (var index=0; index < info.length; index++) {
        this._add_item_attribute(proxy, index);
    }

    return proxy;
};

jigna.ProxyFactory.prototype._update_list_proxy = function(proxy, info) {
    /* Update the given proxy.
     *
     * This removes all previous items and then repopulates the proxy with
     * items that reflect the (possibly) new length.
     */
    this._delete_list_items(proxy);
    this._populate_list_proxy(proxy, info);

    // Get rid of any cached items (items we have already requested from the
    // server-side.
    proxy.__cache__ = []
};

// Common for list and dict proxies ////////////////////////////////////////////

jigna.ProxyFactory.prototype._add_item_attribute = function(proxy, index){
    var descriptor, get, set;

    get = function() {
        // In here, 'this' refers to the proxy!
        var value = this.__cache__[index];
        if (value === undefined) {
            value = this.__client__.get_attribute(this, index);
            this.__cache__[index] = value;
        }

        return value;
    };

    set = function(value) {
        // In here, 'this' refers to the proxy!
        this.__cache__[index] = value;
        this.__client__.set_item(this.__id__, index, value);
    };

    descriptor = {enumerable:true, get:get, set:set, configurable:true};
    Object.defineProperty(proxy, index, descriptor);
};


/////////////////////////////////////////////////////////////////////////////
// AsyncProxyFactory
/////////////////////////////////////////////////////////////////////////////

jigna._SavedData = function(data) {
    // Used internally to save marshaled data to unmarshal later.
    this.data = data;
};

jigna.AsyncProxyFactory = function(client) {
    jigna.ProxyFactory.call(this, client);
};

jigna.AsyncProxyFactory.prototype = Object.create(
    jigna.ProxyFactory.prototype
);

jigna.AsyncProxyFactory.prototype.constructor = jigna.AsyncProxyFactory

jigna.AsyncProxyFactory.prototype._add_instance_attribute = function(proxy, attribute_name){
    var descriptor, get, set;

    get = function() {
        // In here, 'this' refers to the proxy!
        var value = this.__cache__[attribute_name];
        if (value === undefined) {
            value = this.__client__.get_attribute(this, attribute_name);
            if (value === undefined) {
                var info = this.__info__;
                if (info && (info.attribute_values !== undefined)) {
                    var index = info.attribute_names.indexOf(attribute_name);
                    value = this.__client__._unmarshal(
                        info.attribute_values[index]
                    );
                }
            } else {
                this.__cache__[attribute_name] = value;
            }
        }

        return value;
    };

    set = function(value) {
        // In here, 'this' refers to the proxy!
        //
        // If the proxy is for a 'HasTraits' instance then we don't need
        // to set the cached value here as the value will get updated when
        // we get the corresponding trait event. However, setting the value
        // here means that we can create jigna UIs for non-traits objects - it
        // just means we won't react to external changes to the model(s).
        this.__cache__[attribute_name] = value;
        this.__client__.set_instance_attribute(
            this.__id__, attribute_name, value
        );
    };

    descriptor = {enumerable:true, get:get, set:set, configurable:true};
    Object.defineProperty(proxy, attribute_name, descriptor);
};

jigna.AsyncProxyFactory.prototype._populate_dict_proxy = function(proxy, info) {
    var index, key;
    var values = info.values;

    for (index=0; index < info.keys.length; index++) {
        key = info.keys[index];
        this._add_item_attribute(proxy, key);
        proxy.__cache__[key] = new jigna._SavedData(values.data[index]);
    }
};

jigna.AsyncProxyFactory.prototype._update_dict_proxy = function(proxy, info) {
    var removed = info.removed;
    var cache = proxy.__cache__;
    var key;

    // Add the keys in the added.
    this._populate_dict_proxy(proxy, info.added);

    for (var index=0; index < removed.length; index++) {
        key = removed[index];
        delete cache[key];
        delete proxy[key];
    }
};

jigna.AsyncProxyFactory.prototype._populate_list_proxy = function(proxy, info) {
    /* Populate the items in a list proxy. */

    var data = info.data;
    for (var index=0; index < info.length; index++) {
        this._add_item_attribute(proxy, index);
        proxy.__cache__[index] = new jigna._SavedData(data[index]);
    }

    return proxy;
};

jigna.AsyncProxyFactory.prototype._update_list_proxy = function(proxy, info) {
    /* Update the given proxy. */

    if (info.index === undefined) {
        // This is an extended slice.  Note that one cannot increase the size
        // of the list with an extended slice.  So one is either deleting
        // elements or changing them.
        var index;
        var added = info.added;
        var removed = info.removed - added.length;
        var cache = proxy.__cache__;
        var end = cache.length;
        if (removed > 0) {
            // Delete the proxy indices at the end.
            for (index=end; index > (end-removed) ; index--) {
                delete proxy[index-1];
            }
            var to_remove = [];
            for (index=info.start; index<info.stop; index+=info.step) {
                to_remove.push(index);
            }
            // Delete the cached entries in sequence from the back.
            for (index=to_remove.length; index > 0; index--) {
                cache.splice(to_remove[index-1], 1);
            }
        } else {
            // When nothing is removed, just update the cache entries.
            for (var i=0; i < added.length; i++) {
                index = info.start + i*info.step;
                cache[index] = new jigna._SavedData(added.data[i]);
            }
        }
    } else {
        // This is not an extended slice.
        var splice_args = [info.index, info.removed].concat(
            info.added.data.map(function(x) {return new jigna._SavedData(x);})
        );

        var extra = splice_args.length - 2 - splice_args[1];
        var cache = proxy.__cache__;
        var end = cache.length;
        if (extra < 0) {
            for (var index=end; index > (end+extra) ; index--) {
                delete proxy[index-1];
            }
        } else {
            for (var index=0; index < extra; index++){
                this._add_item_attribute(proxy, end+index);
            }
        }
        cache.splice.apply(cache, splice_args);
    }
};


// Common for list and dict proxies ////////////////////////////////////////////

jigna.AsyncProxyFactory.prototype._add_item_attribute = function(proxy, index){
    var descriptor, get, set;

    get = function() {
        // In here, 'this' refers to the proxy!
        var value = this.__cache__[index];
        if (value === undefined) {
            value = this.__client__.get_attribute(this, index);
            this.__cache__[index] = value;
        } else if (value instanceof jigna._SavedData) {
            value = this.__client__._unmarshal(value.data);
            this.__cache__[index] = value;
        }

        return value;
    };

    set = function(value) {
        // In here, 'this' refers to the proxy!
        this.__cache__[index] = value;
        this.__client__.set_item(this.__id__, index, value);
    };

    descriptor = {enumerable:true, get:get, set:set, configurable:true};
    Object.defineProperty(proxy, index, descriptor);
};


///////////////////////////////////////////////////////////////////////////////
// Proxy
///////////////////////////////////////////////////////////////////////////////

jigna.Proxy = function(type, id, client) {
    // We use the '__attribute__' pattern to reduce the risk of name clashes
    // with the actuall attribute and methods on the object that we are a
    // proxy for.
    Object.defineProperty(this, '__type__',   {value : type});
    Object.defineProperty(this, '__id__',     {value : id});
    Object.defineProperty(this, '__client__', {value : client});
    Object.defineProperty(this, '__cache__',  {value : {}, writable: true});

    // The state for each attribute can be 'busy' or undefined, if 'busy' it
    // implies that the server is waiting to receive the value.
    Object.defineProperty(this, '__state__',  {value : {}});
};


// SubArray.js ////////////////////////////////////////////////////////////////
// (C) Copyright Juriy Zaytsev
// Source: 1. https://github.com/kangax/array_subclassing
//         2. http://perfectionkills.com/how-ecmascript-5-still-does-not-allow-
//            to-subclass-an-array/
///////////////////////////////////////////////////////////////////////////////

var makeSubArray = (function(){

    var MAX_SIGNED_INT_VALUE = Math.pow(2, 32) - 1,
        hasOwnProperty = Object.prototype.hasOwnProperty;

    function ToUint32(value) {
        return value >>> 0;
    }

    function getMaxIndexProperty(object) {
        var maxIndex = -1, isValidProperty;

        for (var prop in object) {

            // int conversion of the property
            int_prop = ToUint32(prop);

            isValidProperty = (
                String(int_prop) === prop &&
                int_prop !== MAX_SIGNED_INT_VALUE &&
                hasOwnProperty.call(object, prop)
            );

            if (isValidProperty && int_prop > maxIndex) {
                maxIndex = prop;
            }
        }
        return maxIndex;
    }

    return function(methods) {
        var length = 0;
        methods = methods || { };

        methods.length = {
            get: function() {
                var maxIndexProperty = +getMaxIndexProperty(this);
                return Math.max(length, maxIndexProperty + 1);
            },
            set: function(value) {
                var constrainedValue = ToUint32(value);
                if (constrainedValue !== +value) {
                    throw new RangeError();
                }
                for (var i = constrainedValue, len = this.length; i < len; i++) {
                    delete this[i];
                }
                length = constrainedValue;
            }
        };

        methods.toString = {
            value: Array.prototype.join
        };

        return Object.create(Array.prototype, methods);
    };
})();

jigna.SubArray = function() {
    var arr = makeSubArray();

    if (arguments.length === 1) {
        arr.length = arguments[0];
    }
    else {
        arr.push.apply(arr, arguments);
    }
    return arr;
};


///////////////////////////////////////////////////////////////////////////////
// ListProxy
///////////////////////////////////////////////////////////////////////////////

// ListProxy is handled separately because it has to do special handling
// to behave as regular Javascript `Array` objects
// See "Wrappers. Prototype chain injection" section in this article:
// http://perfectionkills.com/how-ecmascript-5-still-does-not-allow-to-subclass-an-array/

jigna.ListProxy = function(type, id, client) {

    var arr = new jigna.SubArray();

    // fixme: repetition of property definition
    Object.defineProperty(arr, '__type__',   {value : type});
    Object.defineProperty(arr, '__id__',     {value : id});
    Object.defineProperty(arr, '__client__', {value : client});
    Object.defineProperty(arr, '__cache__',  {value : [], writable: true});

    // The state for each attribute can be 'busy' or undefined, if 'busy' it
    // implies that the server is waiting to receive the value.
    Object.defineProperty(arr, '__state__',  {value : {}});

    return arr;
};


///////////////////////////////////////////////////////////////////////////////
// QtBridge (intra-process)
///////////////////////////////////////////////////////////////////////////////

jigna.QtBridge = function(client, qt_bridge) {
    this.ready = new $.Deferred();

    // Private protocol
    this._client    = client;
    this._qt_bridge = qt_bridge;

    this.ready.resolve();
};

jigna.QtBridge.prototype.handle_event = function(jsonized_event) {
    /* Handle an event from the server. */
    this._client.handle_event(jsonized_event);
};

jigna.QtBridge.prototype.send_request = function(jsonized_request) {
    /* Send a request to the server and wait for the reply. */

    result = this._qt_bridge.handle_request(jsonized_request);

    return result;
};

jigna.QtBridge.prototype.send_request_async = function(jsonized_request) {
    /* A dummy async version of the send_request method. Since QtBridge is
    single process, this method indeed waits for the reply but presents
    a deferred API so that the AsyncClient can use it. Mainly for testing
    purposes only. */

    var deferred = new $.Deferred();

    deferred.resolve(this._qt_bridge.handle_request(jsonized_request));

    return deferred.promise();
};


///////////////////////////////////////////////////////////////////////////////
// WebBridge
///////////////////////////////////////////////////////////////////////////////

jigna.WebBridge = function(client) {
    this._client = client;

    // The jigna_server attribute can be set by a client to point to a
    // different Jigna server.
    var jigna_server = window['jigna_server'];
    if (jigna_server === undefined) {
        jigna_server = window.location.host;
    }
    this._server_url = 'http://' + jigna_server;

    var url = 'ws://' + jigna_server + '/_jigna_ws';

    this._deferred_requests = {};
    this._request_ids = [];
    for (var index=0; index < 1024; index++) {
        this._request_ids.push(index);
    }

    this._web_socket = new WebSocket(url);
    this.ready = new $.Deferred();
    var bridge = this;
    this._web_socket.onopen = function() {
        bridge.ready.resolve();
    };
    this._web_socket.onmessage = function(event) {
        bridge.handle_event(event.data);
    };
};

jigna.WebBridge.prototype.handle_event = function(jsonized_event) {
    /* Handle an event from the server. */
    var response = JSON.parse(jsonized_event);
    var request_id = response[0];
    var jsonized_response = response[1];
    if (request_id === -1) {
        this._client.handle_event(jsonized_response);
    }
    else {
        var deferred = this._pop_deferred_request(request_id);
        deferred.resolve(jsonized_response);
    }
};

jigna.WebBridge.prototype.send_request = function(jsonized_request) {
    /* Send a request to the server and wait for the reply. */

    var jsonized_response;

    $.ajax(
        {
            url     : '/_jigna',
            type    : 'GET',
            data    : {'data': jsonized_request},
            success : function(result) {jsonized_response = result;},
            error   : function(status, error) {
                          console.warning("Error: " + error);
                      },
            async   : false
        }
    );

    return jsonized_response;
};

jigna.WebBridge.prototype.send_request_async = function(jsonized_request) {
    /* Send a request to the server and do not wait and return a Promise
       which is resolved upon completion of the request.
    */

    var deferred = new $.Deferred();
    var request_id = this._push_deferred_request(deferred);
    var bridge = this;
    this.ready.done(function() {
        bridge._web_socket.send(JSON.stringify([request_id, jsonized_request]));
    });
    return deferred.promise();
};

//// Private protocol /////////////////////////////////////////////////////

jigna.WebBridge.prototype._pop_deferred_request = function(request_id) {
    var deferred = this._deferred_requests[request_id];
    delete this._deferred_requests[request_id];
    this._request_ids.push(request_id);
    return deferred;
};

jigna.WebBridge.prototype._push_deferred_request = function(deferred) {
    var id = this._request_ids.pop();
    if (id === undefined) {
        console.error("In _push_deferred_request, request_id is undefined.");
    }
    this._deferred_requests[id] = deferred;
    return id;
};


// A Horrible hack to update objects.  This was gleaned from the vuejs
// code.  The problem we have is that vuejs cannot listen to changes to
// model changes because we use getters/setters.  Internally vue uses an
// observer to notify dependent elements.  We use the __ob__ attribute
// to get the observer and call its `dep.notify()`, this makes
// everything work really well.
jigna.add_listener('jigna', 'object_changed', function (event) {
    var obj = event.object;
    if (obj && obj.__ob__) {
        obj.__ob__.dep.notify();
    }
});


window.$ = $;
window.Vue = Vue;
window.jigna = jigna;

})();
