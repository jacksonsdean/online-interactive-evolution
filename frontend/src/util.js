export function string_to_ids(a){
    var b = a.split(',').map(function(item) {
        return parseInt(item, 10);
    });
    return b;
}

export function ids_to_string(a){
    var b = a.map(function(item) {
        return item.toString();
    });
    return b.join(',');
}

export function post(url, data) {
    return fetch(url, {method: "POST", headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
}