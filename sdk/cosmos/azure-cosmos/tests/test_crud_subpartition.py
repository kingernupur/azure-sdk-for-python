# -*- coding: utf-8 -*-
# The MIT License (MIT)
# Copyright (c) Microsoft Corporation. All rights reserved.

"""End-to-end test.
"""

import time
import unittest
import uuid

import pytest
import requests
from azure.core.pipeline.transport import RequestsTransport, RequestsTransportResponse

import azure.cosmos.cosmos_client as cosmos_client
import azure.cosmos.documents as documents
import azure.cosmos.exceptions as exceptions
import test_config
from azure.cosmos import _retry_utility
from azure.cosmos._routing import routing_range
from azure.cosmos._routing.collection_routing_map import CollectionRoutingMap
from azure.cosmos.http_constants import HttpHeaders, StatusCodes, SubStatusCodes
from azure.cosmos.partition_key import PartitionKey


class TimeoutTransport(RequestsTransport):
    def __init__(self, response):
        self._response = response
        super(TimeoutTransport, self).__init__()

    def send(self, *args, **kwargs):
        if kwargs.pop("passthrough", False):
            return super(TimeoutTransport, self).send(*args, **kwargs)

        time.sleep(5)
        if isinstance(self._response, Exception):
            raise self._response
        output = requests.Response()
        output.status_code = self._response
        response = RequestsTransportResponse(None, output)
        return response

@pytest.mark.cosmosLong
class TestSubpartitionCrud(unittest.TestCase):
    """Python CRUD Tests.
    """
    configs = test_config.TestConfig
    host = configs.host
    masterKey = configs.masterKey
    connectionPolicy = configs.connectionPolicy
    last_headers = []
    client: cosmos_client.CosmosClient = None

    def __AssertHTTPFailureWithStatus(self, status_code, func, *args, **kwargs):
        """Assert HTTP failure with status.

        :Parameters:
            - `status_code`: int
            - `func`: function
        """
        try:
            func(*args, **kwargs)
            self.assertFalse(True, 'function should fail.')
        except exceptions.CosmosHttpResponseError as inst:
            self.assertEqual(inst.status_code, status_code)

    @classmethod
    def setUpClass(cls):
        if (cls.masterKey == '[YOUR_KEY_HERE]' or
                cls.host == '[YOUR_ENDPOINT_HERE]'):
            raise Exception(
                "You must specify your Azure Cosmos account values for "
                "'masterKey' and 'host' at the top of this class to run the "
                "tests.")
        cls.client = cosmos_client.CosmosClient(cls.host, cls.masterKey)
        cls.databaseForTest = cls.client.get_database_client(cls.configs.TEST_DATABASE_ID)

    def test_collection_crud_subpartition(self):
        created_db = self.databaseForTest
        collections = list(created_db.list_containers())
        # create a collection
        before_create_collections_count = len(collections)
        collection_id = 'test_collection_crud_MH ' + str(uuid.uuid4())
        collection_indexing_policy = {'indexingMode': 'consistent'}
        created_collection = created_db.create_container(id=collection_id,
                                                         indexing_policy=collection_indexing_policy,
                                                         partition_key=PartitionKey(path=["/pk1", "/pk2", "/pk3"],
                                                                                    kind="MultiHash"))
        self.assertEqual(collection_id, created_collection.id)

        created_properties = created_collection.read()
        self.assertEqual('consistent', created_properties['indexingPolicy']['indexingMode'])

        # read collections after creation
        collections = list(created_db.list_containers())
        self.assertEqual(len(collections),
                         before_create_collections_count + 1,
                         'create should increase the number of collections')
        # query collections
        collections = list(created_db.query_containers(
            {
                'query': 'SELECT * FROM root r WHERE r.id=@id',
                'parameters': [
                    {'name': '@id', 'value': collection_id}
                ]
            }))

        self.assertTrue(collections)
        # delete collection
        created_db.delete_container(created_collection.id)
        # read collection after deletion
        created_container = created_db.get_container_client(created_collection.id)
        self.__AssertHTTPFailureWithStatus(StatusCodes.NOT_FOUND,
                                           created_container.read)

        container_proxy = created_db.create_container(id=created_collection.id,
                                                      partition_key=PartitionKey(path=
                                                                                 ["/id1", "/id2", "/id3"],
                                                                                 kind='MultiHash'))
        self.assertEqual(created_collection.id, container_proxy.id)
        container_properties = container_proxy._get_properties()
        self.assertDictEqual(PartitionKey(path=["/id1", "/id2", "/id3"], kind='MultiHash'),
                             container_properties['partitionKey'])

        created_db.delete_container(created_collection.id)

    def test_partitioned_collection_subpartition(self):
        created_db = self.databaseForTest

        collection_definition = {'id': 'test_partitioned_collection_MH ' + str(uuid.uuid4()),
                                 'partitionKey':
                                     {
                                         'paths': ['/id', '/pk'],
                                         'kind': documents.PartitionKind.MultiHash,
                                         'version': 2
                                     }
                                 }

        offer_throughput = 10100
        created_collection = created_db.create_container(id=collection_definition['id'],
                                                         partition_key=collection_definition['partitionKey'],
                                                         offer_throughput=offer_throughput)

        self.assertEqual(collection_definition.get('id'), created_collection.id)

        created_collection_properties = created_collection.read()
        self.assertEqual(collection_definition.get('partitionKey').get('paths'),
                         created_collection_properties['partitionKey']['paths'])
        self.assertEqual(collection_definition.get('partitionKey').get('kind'),
                         created_collection_properties['partitionKey']['kind'])

        expected_offer = created_collection.get_throughput()

        self.assertIsNotNone(expected_offer)

        self.assertEqual(expected_offer.offer_throughput, offer_throughput)

        # Negative test, check that user can't make a subpartition higher than 3 levels
        collection_definition2 = {'id': 'test_partitioned_collection2_MH ' + str(uuid.uuid4()),
                                  'partitionKey':
                                      {
                                          'paths': ['/id', '/pk', '/id2', "/pk2"],
                                          'kind': documents.PartitionKind.MultiHash,
                                          'version': 2
                                      }
                                  }
        try:
            created_collection = created_db.create_container(id=collection_definition['id'],
                                                             partition_key=collection_definition2['partitionKey'],
                                                             offer_throughput=offer_throughput)
        except exceptions.CosmosHttpResponseError as error:
            self.assertEqual(error.status_code, StatusCodes.BAD_REQUEST)
            self.assertTrue("Too many partition key paths" in error.message)

        # Negative Test: Check if user tries to create multihash container while defining single hash
        collection_definition3 = {'id': 'test_partitioned_collection2_MH ' + str(uuid.uuid4()),
                                  'partitionKey':
                                      {
                                          'paths': ['/id', '/pk', '/id2', "/pk2"],
                                          'kind': documents.PartitionKind.Hash,
                                          'version': 2
                                      }
                                  }
        try:
            created_collection = created_db.create_container(id=collection_definition['id'],
                                                             partition_key=collection_definition3['partitionKey'],
                                                             offer_throughput=offer_throughput)
        except exceptions.CosmosHttpResponseError as error:
            self.assertEqual(error.status_code, StatusCodes.BAD_REQUEST)
            self.assertTrue("Too many partition key paths" in error.message)
        created_db.delete_container(created_collection.id)

    def test_partitioned_collection_partition_key_extraction_subpartition(self):
        created_db = self.databaseForTest

        collection_id = 'test_partitioned_collection_partition_key_extraction_MH ' + str(uuid.uuid4())
        created_collection = created_db.create_container(
            id=collection_id,
            partition_key=PartitionKey(path=['/address/state', '/address/city'], kind=documents.PartitionKind.MultiHash)
        )

        document_definition = {'id': 'document1',
                               'address': {'street': '1 Microsoft Way',
                                           'city': 'Redmond',
                                           'state': 'WA',
                                           'zip code': 98052
                                           }
                               }

        self.OriginalExecuteFunction = _retry_utility.ExecuteFunction
        _retry_utility.ExecuteFunction = self._MockExecuteFunction
        # create document without partition key being specified
        created_document = created_collection.create_item(body=document_definition)
        _retry_utility.ExecuteFunction = self.OriginalExecuteFunction
        self.assertEqual(self.last_headers[0], '["WA","Redmond"]')
        del self.last_headers[:]

        self.assertEqual(created_document.get('id'), document_definition.get('id'))
        self.assertEqual(created_document.get('address').get('state'), document_definition.get('address').get('state'))

        collection_id = 'test_partitioned_collection_partition_key_extraction_MH_2 ' + str(uuid.uuid4())
        created_collection2 = created_db.create_container(
            id=collection_id,
            partition_key=PartitionKey(path=['/address/state/city', '/address/city/state'],
                                       kind=documents.PartitionKind.MultiHash)
        )

        self.OriginalExecuteFunction = _retry_utility.ExecuteFunction
        _retry_utility.ExecuteFunction = self._MockExecuteFunction
        # Create document with partition key not present in the document
        try:
            created_document = created_collection2.create_item(document_definition)
            _retry_utility.ExecuteFunction = self.OriginalExecuteFunction
            del self.last_headers[:]
            self.fail('Operation Should Fail.')
        except exceptions.CosmosHttpResponseError as error:
            self.assertEqual(error.status_code, StatusCodes.BAD_REQUEST)
            self.assertEqual(error.sub_status, SubStatusCodes.PARTITION_KEY_MISMATCH)
            del self.last_headers[:]

        created_db.delete_container(created_collection.id)
        created_db.delete_container(created_collection2.id)

    def test_partitioned_collection_partition_key_extraction_special_chars_subpartition(self):
        created_db = self.databaseForTest

        collection_id = 'test_partitioned_collection_partition_key_extraction_special_chars_MH_1 ' + str(uuid.uuid4())

        created_collection1 = created_db.create_container(
            id=collection_id,
            partition_key=PartitionKey(path=['/\"first level\' 1*()\"/\"le/vel2\"',
                                             '/\"second level\' 1*()\"/\"le/vel2\"'],
                                       kind=documents.PartitionKind.MultiHash)
        )
        document_definition = {'id': 'document1',
                               "first level' 1*()": {"le/vel2": 'val1'},
                               "second level' 1*()": {"le/vel2": 'val2'}
                               }
        self.OriginalExecuteFunction = _retry_utility.ExecuteFunction
        _retry_utility.ExecuteFunction = self._MockExecuteFunction
        created_document = created_collection1.create_item(body=document_definition)
        _retry_utility.ExecuteFunction = self.OriginalExecuteFunction
        self.assertEqual(self.last_headers[-1], '["val1","val2"]')
        del self.last_headers[:]

        collection_definition2 = {
            'id': 'test_partitioned_collection_partition_key_extraction_special_chars_MH_2 ' + str(uuid.uuid4()),
            'partitionKey':
                {
                    'paths': ['/\'first level\" 1*()\'/\'first le/vel2\'',
                              '/\'second level\" 1*()\'/\'second le/vel2\''],
                    'kind': documents.PartitionKind.MultiHash
                }
        }

        created_collection2 = created_db.create_container(
            id=collection_definition2['id'],
            partition_key=PartitionKey(path=collection_definition2["partitionKey"]["paths"]
                                       , kind=collection_definition2["partitionKey"]["kind"])
        )

        document_definition = {'id': 'document2',
                               'first level\" 1*()': {'first le/vel2': 'val3'},
                               'second level\" 1*()': {'second le/vel2': 'val4'}
                               }

        self.OriginalExecuteFunction = _retry_utility.ExecuteFunction
        _retry_utility.ExecuteFunction = self._MockExecuteFunction
        # create document without partition key being specified
        created_document = created_collection2.create_item(body=document_definition)
        _retry_utility.ExecuteFunction = self.OriginalExecuteFunction
        self.assertEqual(self.last_headers[-1], '["val3","val4"]')
        del self.last_headers[:]

        created_db.delete_container(created_collection1.id)
        created_db.delete_container(created_collection2.id)

    def test_partitioned_collection_document_crud_and_query_subpartition(self):
        created_db = self.databaseForTest

        collection_id = 'test_partitioned_collection_partition_document_crud_and_query_MH ' + str(uuid.uuid4())
        created_collection = created_db.create_container(
            id=collection_id,
            partition_key=PartitionKey(path=['/city', '/zipcode'], kind=documents.PartitionKind.MultiHash)
        )

        document_definition = {'id': 'document',
                               'key': 'value',
                               'city': 'Redmond',
                               'zipcode': '98052'}

        created_document = created_collection.create_item(
            body=document_definition
        )

        self.assertEqual(created_document.get('id'), document_definition.get('id'))
        self.assertEqual(created_document.get('key'), document_definition.get('key'))
        self.assertEqual(created_document.get('city'), document_definition.get('city'))
        self.assertEqual(created_document.get('zipcode'), document_definition.get('zipcode'))

        # read document
        read_document = created_collection.read_item(
            item=created_document.get('id'),
            partition_key=[created_document.get('city'), created_document.get('zipcode')]
        )

        self.assertEqual(read_document.get('id'), created_document.get('id'))
        self.assertEqual(read_document.get('key'), created_document.get('key'))
        self.assertEqual(read_document.get('city'), created_document.get('city'))
        self.assertEqual(read_document.get('zipcode'), created_document.get('zipcode'))

        # Read document feed doesn't require partitionKey as it's always a cross partition query
        documentlist = list(created_collection.read_all_items())
        self.assertEqual(1, len(documentlist))

        # replace document
        document_definition['key'] = 'new value'

        replaced_document = created_collection.replace_item(
            item=read_document,
            body=document_definition
        )

        self.assertEqual(replaced_document.get('key'), document_definition.get('key'))

        # upsert document(create scenario)
        document_definition['id'] = 'document2'
        document_definition['key'] = 'value2'
        document_definition['city'] = 'Atlanta'
        document_definition['zipcode'] = '30363'

        upserted_document = created_collection.upsert_item(body=document_definition)

        self.assertEqual(upserted_document.get('id'), document_definition.get('id'))
        self.assertEqual(upserted_document.get('key'), document_definition.get('key'))
        self.assertEqual(upserted_document.get('city'), document_definition.get('city'))
        self.assertEqual(upserted_document.get('zipcode'), document_definition.get('zipcode'))

        documentlist = list(created_collection.read_all_items())
        self.assertEqual(2, len(documentlist))

        # delete document
        created_collection.delete_item(item=upserted_document, partition_key=[upserted_document.get('city'),
                                                                              upserted_document.get('zipcode')])

        # query document on the partition key specified in the predicate will pass even without setting
        # enableCrossPartitionQuery or passing in the partitionKey value
        documentlist = list(created_collection.query_items(
            {
                'query': 'SELECT * FROM root r WHERE r.city=\'' + replaced_document.get(
                    'city') + '\' and r.zipcode=\'' + replaced_document.get('zipcode') + '\''
                # pylint: disable=line-too-long
            }))
        self.assertEqual(1, len(documentlist))

        # query document on any property other than partitionKey will fail without setting enableCrossPartitionQuery
        # or passing in the partitionKey value
        try:
            list(created_collection.query_items(
                {
                    'query': 'SELECT * FROM root r WHERE r.key=\'' + replaced_document.get('key') + '\''  # nosec
                }))
        except Exception:
            pass

        # cross partition query
        documentlist = list(created_collection.query_items(
            query='SELECT * FROM root r WHERE r.key=\'' + replaced_document.get('key') + '\'',  # nosec
            enable_cross_partition_query=True
        ))

        self.assertEqual(1, len(documentlist))

        # query document by providing the partitionKey value
        documentlist = list(created_collection.query_items(
            query='SELECT * FROM root r WHERE r.key=\'' + replaced_document.get('key') + '\'',  # nosec
            partition_key=[replaced_document.get('city'), replaced_document.get('zipcode')]
        ))

        self.assertEqual(1, len(documentlist))

        # Using incomplete extracted partition key in item body
        incomplete_document = {'id': 'document3',
                               'key': 'value3',
                               'city': 'Vancouver'}

        try:
            created_collection.create_item(body=incomplete_document)
            self.fail("Test did not fail as expected")
        except exceptions.CosmosHttpResponseError as error:
            self.assertEqual(error.status_code, StatusCodes.BAD_REQUEST)
            self.assertEqual(error.sub_status, SubStatusCodes.PARTITION_KEY_MISMATCH)

        # using incomplete partition key in read item
        try:
            created_collection.read_item(created_document, partition_key=["Redmond"])
            self.fail("Test did not fail as expected")
        except exceptions.CosmosHttpResponseError as error:
            self.assertEqual(error.status_code, StatusCodes.BAD_REQUEST)
            self.assertEqual(error.sub_status, SubStatusCodes.PARTITION_KEY_MISMATCH)

        # using mix value types for partition key
        doc_mixed_types = {'id': "doc4",
                           'key': 'value4',
                           'city': None,
                           'zipcode': 1000}
        created_mixed_type_doc = created_collection.create_item(body=doc_mixed_types)
        self.assertEqual(doc_mixed_types.get('city'), created_mixed_type_doc.get('city'))
        self.assertEqual(doc_mixed_types.get('zipcode'), created_mixed_type_doc.get('zipcode'))

        created_db.delete_container(collection_id)

    def test_partitioned_collection_prefix_partition_query_subpartition(self):
        created_db = self.databaseForTest

        collection_id = 'test_partitioned_collection_partition_key_prefix_query_MH ' + str(uuid.uuid4())
        created_collection = created_db.create_container(
            id=collection_id,
            partition_key=PartitionKey(path=['/state', '/city', '/zipcode'], kind=documents.PartitionKind.MultiHash)
        )

        item_values = [
            ["CA", "Newbury Park", "91319"],
            ["CA", "Oxnard", "93033"],
            ["CA", "Oxnard", "93030"],
            ["CA", "Oxnard", "93036"],
            ["CA", "Thousand Oaks", "91358"],
            ["CA", "Ventura", "93002"],
            ["CA", "Ojai", "93023"],  # cspell:disable-line
            ["CA", "Port Hueneme", "93041"],  # cspell:disable-line
            ["WA", "Seattle", "98101"],
            ["WA", "Bellevue", "98004"]
        ]

        document_definitions = [{'id': 'document1',
                                 'state': item_values[0][0],
                                 'city': item_values[0][1],
                                 'zipcode': item_values[0][2]
                                 },
                                {'id': 'document2',
                                 'state': item_values[1][0],
                                 'city': item_values[1][1],
                                 'zipcode': item_values[1][2]
                                 },
                                {'id': 'document3',
                                 'state': item_values[2][0],
                                 'city': item_values[2][1],
                                 'zipcode': item_values[2][2]
                                 },
                                {'id': 'document4',
                                 'state': item_values[3][0],
                                 'city': item_values[3][1],
                                 'zipcode': item_values[3][2]
                                 },
                                {'id': 'document5',
                                 'state': item_values[4][0],
                                 'city': item_values[4][1],
                                 'zipcode': item_values[4][2]
                                 },
                                {'id': 'document6',
                                 'state': item_values[5][0],
                                 'city': item_values[5][1],
                                 'zipcode': item_values[5][2]
                                 },
                                {'id': 'document7',
                                 'state': item_values[6][0],
                                 'city': item_values[6][1],
                                 'zipcode': item_values[6][2]
                                 },
                                {'id': 'document8',
                                 'state': item_values[7][0],
                                 'city': item_values[7][1],
                                 'zipcode': item_values[7][2]
                                 },
                                {'id': 'document9',
                                 'state': item_values[8][0],
                                 'city': item_values[8][1],
                                 'zipcode': item_values[8][2]
                                 },
                                {'id': 'document10',
                                 'state': item_values[9][0],
                                 'city': item_values[9][1],
                                 'zipcode': item_values[9][2]
                                 }
                                ]
        created_documents = []
        for document_definition in document_definitions:
            created_documents.append(created_collection.create_item(
                body=document_definition))
        self.assertEqual(len(created_documents), len(document_definitions))

        # Query all documents should return all items
        document_list = list(created_collection.query_items(query='Select * from c', enable_cross_partition_query=True))
        self.assertEqual(len(document_list), len(document_definitions))

        # Query all items with only CA for 1st level. Should return only 8 items instead of 10
        document_list = list(created_collection.query_items(query='Select * from c', partition_key=['CA']))
        self.assertEqual(8, len(document_list))

        # Query all items with CA for 1st level and Oxnard for second level. Should only return 3 items
        document_list = list(created_collection.query_items(query='Select * from c', partition_key=['CA', 'Oxnard']))
        self.assertEqual(3, len(document_list))

        # Query for specific zipcode using 1st level of partition key value only:
        document_list = list(created_collection.query_items(query='Select * from c where c.zipcode = "93033"',
                                                            partition_key=['CA']))
        self.assertEqual(1, len(document_list))

        # Query Should work with None values:
        document_list = list(created_collection.query_items(query='Select * from c', partition_key=[None, '93033']))
        self.assertEqual(0, len(document_list))

        # Query Should Work with non string values
        document_list = list(created_collection.query_items(query='Select * from c', partition_key=[0xFF, 0xFF]))
        self.assertEqual(0, len(document_list))

        document_list = list(created_collection.query_items(query='Select * from c', partition_key=[None, None]))
        self.assertEqual(0, len(document_list))

        document_list = list(created_collection.query_items(query='Select * from c', partition_key=["", ""]))
        self.assertEqual(0, len(document_list))

        document_list = list(created_collection.query_items(query='Select * from c', partition_key=[""]))
        self.assertEqual(0, len(document_list))

        # Negative Test, prefix query should not work if no partition is given (empty list is given)
        try:
            document_list = list(created_collection.query_items(query='Select * from c', partition_key=[]))
            self.fail("Test did not fail as expected")
        except exceptions.CosmosHttpResponseError as error:
            self.assertEqual(error.status_code, StatusCodes.BAD_REQUEST)
            self.assertTrue("Cross partition query is required but disabled"
                            in error.message)

    def test_partition_key_range_overlap_subpartition(self):
        Id = 'id'
        MinInclusive = 'minInclusive'
        MaxExclusive = 'maxExclusive'
        partitionKeyRanges = \
            [
                ({Id: "2",
                  MinInclusive: "0000000050",
                  MaxExclusive: "0000000070"},
                 2),
                ({Id: "0",
                  MinInclusive: "",
                  MaxExclusive: "0000000030"},
                 0),
                ({Id: "1",
                  MinInclusive: "0000000030",
                  MaxExclusive: "0000000050"},
                 1),
                ({Id: "3",
                  MinInclusive: "0000000070",
                  MaxExclusive: "FF"},
                 3)
            ]

        crm = CollectionRoutingMap.CompleteRoutingMap(partitionKeyRanges, "")

        # Case 1: EPK range matches a single entire physical partition
        EPK_range_1 = routing_range.Range(range_min="0000000030", range_max="0000000050",
                                          isMinInclusive=True, isMaxInclusive=False)
        over_lapping_ranges_1 = crm.get_overlapping_ranges([EPK_range_1])
        # Should only have 1 over lapping range
        self.assertEqual(len(over_lapping_ranges_1), 1)
        # EPK range 1 should be overlapping physical partition 1
        self.assertEqual(over_lapping_ranges_1[0][Id], "1")
        # Partition 1 and EPK range 1 should have same range min and range max
        over_lapping_range_1 = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_1[0])
        self.assertEqual(over_lapping_range_1.min, EPK_range_1.min)
        self.assertEqual(over_lapping_range_1.max, EPK_range_1.max)

        # Case 2: EPK range is a sub range of a single physical partition

        EPK_range_2 = routing_range.Range(range_min="0000000035", range_max="0000000045",
                                          isMinInclusive=True, isMaxInclusive=False)
        over_lapping_ranges_2 = crm.get_overlapping_ranges([EPK_range_2])
        # Should only have 1 over lapping range
        self.assertEqual(len(over_lapping_ranges_2), 1)
        # EPK range 2 should be overlapping physical partition 1
        self.assertEqual(over_lapping_ranges_2[0][Id], "1")
        # EPK range 2 min should be higher than over lapping partition and the max should be lower
        over_lapping_range_2 = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_2[0])
        self.assertLess(over_lapping_range_2.min, EPK_range_2.min)
        self.assertLess(EPK_range_2.max, over_lapping_range_2.max)

        # Case 3: EPK range partially spans 2 physical partitions

        EPK_range_3 = routing_range.Range(range_min="0000000035", range_max="0000000055",
                                          isMinInclusive=True, isMaxInclusive=False)
        over_lapping_ranges_3 = crm.get_overlapping_ranges([EPK_range_3])
        # Should overlap exactly two partition ranges
        self.assertEqual(len(over_lapping_ranges_3), 2)
        # EPK range 3 should be over lapping partition 1 and partition 2
        self.assertEqual(over_lapping_ranges_3[0][Id], "1")
        self.assertEqual(over_lapping_ranges_3[1][Id], "2")
        # EPK Range 3 range min should be higher than partition 1's min, but lower than partition 2's, vice versa with max
        over_lapping_range_3A = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_3[0])
        over_lapping_range_3B = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_3[1])
        self.assertLess(over_lapping_range_3A.min, EPK_range_3.min)
        self.assertLess(EPK_range_3.min, over_lapping_range_3B.min)
        self.assertGreater(EPK_range_3.max, over_lapping_range_3A.max)
        self.assertGreater(over_lapping_range_3B.max, EPK_range_3.max)

        # Case 4: EPK range spans multiple physical partitions, including entire physical partitions

        EPK_range_4 = routing_range.Range(range_min="0000000020", range_max="0000000060",
                                          isMinInclusive=True, isMaxInclusive=False)
        over_lapping_ranges_4 = crm.get_overlapping_ranges([EPK_range_4])
        # should overlap 3 partitions
        self.assertEqual(len(over_lapping_ranges_4), 3)
        # EPK range 4 should be over lapping partitions 0, 1, and 2
        self.assertEqual(over_lapping_ranges_4[0][Id], "0")
        self.assertEqual(over_lapping_ranges_4[1][Id], "1")
        self.assertEqual(over_lapping_ranges_4[2][Id], "2")

        # individual ranges for each partition
        olr_4_a = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_4[0])
        olr_4_b = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_4[1])
        olr_4_c = routing_range.Range.PartitionKeyRangeToRange(over_lapping_ranges_4[2])
        # both EPK range 4 min and max should be greater than partitions 0 min and max
        self.assertGreater(EPK_range_4.min, olr_4_a.min)
        self.assertGreater(EPK_range_4.max, olr_4_a.max)
        # EPK range 4 should contain partition 1's range entirely
        self.assertTrue(EPK_range_4.contains(olr_4_b.min))
        self.assertTrue(EPK_range_4.contains(olr_4_b.max))
        # Both EPK range 4 min and max should be less than partition 2's min and max
        self.assertLess(EPK_range_4.min, olr_4_c.min)
        self.assertLess(EPK_range_4.max, olr_4_c.max)

    def test_partitioned_collection_query_with_tuples_subpartition(self):
        created_db = self.databaseForTest

        collection_id = 'test_partitioned_collection_query_with_tuples_MH ' + str(uuid.uuid4())
        created_collection = created_db.create_container(
            id=collection_id,
            partition_key=PartitionKey(path=['/state', '/city', '/zipcode'], kind=documents.PartitionKind.MultiHash)
        )

        document_definition = {'id': 'document1',
                               'state': 'CA',
                               'city': 'Oxnard',
                               'zipcode': '93033'}

        created_document = created_collection.create_item(body=document_definition)
        self.assertEqual(created_document.get('id'), document_definition.get('id'))

        # Query using tuple instead of list
        document_list = list(
            created_collection.query_items(query='Select * from c', partition_key=('CA', 'Oxnard', '93033')))
        self.assertEqual(1, len(document_list))

        created_db.delete_container(created_collection.id)

    # Commenting out delete items by pk until test pipelines support it
    # def test_delete_all_items_by_partition_key_subpartition(self):
    #     # create database
    #     created_db = self.databaseForTest
    #
    #     # create container
    #     created_collection = created_db.create_container(
    #         id='test_delete_all_items_by_partition_key ' + str(uuid.uuid4()),
    #         partition_key=PartitionKey(path=['/pk1','/pk2','/pk3'], kind='MultiHash')
    #     )
    #     # Create two partition keys
    #     partition_key1 = ['pkA1 ' + str(uuid.uuid4()), 'pkA2 ' + str(uuid.uuid4()), 'pkA3 ' + str(uuid.uuid4())]
    #     partition_key2 = ['pkB1 ' + str(uuid.uuid4()), 'pkB2 ' + str(uuid.uuid4()), 'pkB3 ' + str(uuid.uuid4())]
    #
    #     # add items for partition key 1
    #     for i in range(1, 3):
    #         created_collection.upsert_item(
    #             dict(id="item{}".format(i), pk1=partition_key1[0], pk2=partition_key1[1], pk3=partition_key1[2])
    #         )
    #
    #     # add items for partition key 2
    #
    #     pk2_item = created_collection.upsert_item(dict(id="item{}".format(3), pk1=partition_key2[0]
    #                                               , pk2=partition_key2[1], pk3=partition_key2[2]))
    #
    #     # delete all items for partition key 1
    #     created_collection.delete_all_items_by_partition_key(partition_key1)
    #
    #     # check that only items from partition key 1 have been deleted
    #     items = list(created_collection.read_all_items())
    #
    #     # items should only have 1 item and it should equal pk2_item
    #     self.assertDictEqual(pk2_item, items[0])
    #
    #     # attempting to delete a non-existent partition key or passing none should not delete
    #     # anything and leave things unchanged
    #     created_collection.delete_all_items_by_partition_key(None)
    #
    #     # check that no changes were made by checking if the only item is still there
    #     items = list(created_collection.read_all_items())
    #
    #     # items should only have 1 item and it should equal pk2_item
    #     self.assertDictEqual(pk2_item, items[0])
    #
    #     created_db.delete_container(created_collection)

    def _MockExecuteFunction(self, function, *args, **kwargs):
        try:
            self.last_headers.append(args[4].headers[HttpHeaders.PartitionKey]
                                     if HttpHeaders.PartitionKey in args[4].headers else '')
        except IndexError:
            self.last_headers.append('')
        return self.OriginalExecuteFunction(function, *args, **kwargs)


if __name__ == '__main__':
    try:
        unittest.main()
    except SystemExit as inst:
        if inst.args[0] is True:  # raised by sys.exit(True) when tests failed
            raise
